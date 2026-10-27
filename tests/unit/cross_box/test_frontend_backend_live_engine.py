# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import asyncio
import base64
import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import httpx
from aiortc import RTCPeerConnection
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI
from pipecat.frames.frames import InputAudioRawFrame, LLMTextFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.frame_processor import FrameProcessor

from delegation_core.config import CoreSettings
from delegation_core.im.input import BackendTask, compose, settings_for
from examples.frontend_backend_live.adapter.adapter import LIVE_AUDIO_IN_RATE, CascadeAdapter
from examples.frontend_backend_live.cascade.heard_log import HEADER as TRANSCRIPT_HEADER
from examples.frontend_backend_live.cascade.heard_log import Heard
from examples.frontend_backend_live.cascade.lines import catalog_lines
from live.gateway import LiveGateway, LiveSettings, create_live_router
from live.gateway.websocket import run_primary_websocket
from live.ports import SessionParts
from live.protocol import SessionConfig
from live.server_delegation.passthrough import COMPOSED_SOURCE, client_settings
from tests.unit.cascade_helpers import (
    CORE_YAML,
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
from tests.unit.core_helpers import FakeChat, chunk, tool_fragment
from tests.unit.live_helpers import FakeWebSocket

RESPONSES_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {"model": "org/org/model", "instructions": "Look things up.", "tools": [MENU_TOOL]},
    },
}

# What tau2's Live client starts a session with: its own agent model, its policy as the backend's instructions, its
# function tools, one call per round and an effort (live_provider.py:283-291).
TAU2_SESSION = {
    "model": "live-1",
    "instructions": "You are a customer service agent. Delegate every request.",
    "delegation": {
        "type": "responses",
        "responses": {
            "model": "tau2/agent-model",
            "instructions": "Policy.",
            "tools": [MENU_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"effort": "low"},
        },
    },
}
# The benchmark's core profile (unlisted_models: fallback).
CORE_BENCH_YAML = CORE_YAML.with_name("core.bench.yaml")

SILENCE_100MS = base64.b64encode(bytes(4800)).decode()


class EngineHarness(unittest.IsolatedAsyncioTestCase):
    def feed_silence(self, tenths):
        """Send caller audio so the input media clock advances (append acknowledgments wait for it)."""
        for _ in range(tenths):
            self.ws.send({"type": "session.input_audio.append", "audio": SILENCE_100MS})

    async def serve(self, session, script, *, live_config=LIVE_CONFIG, delegation=None, core=CORE_YAML, tts=FakeTTS):
        """Run the real gateway with a real engine, scripted models and fake speech services.

        ``live_config`` is the cascade's block, ``core`` the core's file (a profile), and ``delegation`` sets keys of
        its ``delegation`` block; ``tts`` is the speech output's class.
        """
        self.chat = FakeChat(*script)
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

        # The gateway's one core, as the root builds it: the shipped core.yaml, its backend LLM on the same script.
        self.core = chat_core(self.chat, path=core, **(delegation or {}))
        self.addAsyncCleanup(self.core.aclose)
        hook = core_hook(self.core)
        self.core_client = hook.client
        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60), core=hook)
        self.ws = FakeWebSocket()
        self.server = asyncio.create_task(run_primary_websocket(self.gateway, self.ws))
        self.addAsyncCleanup(self.stop)
        self.ws.send({"type": "session.start", "event_id": "start_1", "session": session})
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
            return settings_for(client_settings(SessionConfig.model_validate(session)), defaults)

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
        apology = catalog_lines().apology
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "again?" in e["delta"])
        self.assertEqual(" ".join(self.tts.spoken), " ".join([*catalog_lines().lead_in.split(), *apology.split()]))
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
        settings = client_settings(self.live_session.config)
        self.assertEqual((settings["instructions"], settings["max_output_tokens"]), ("New backend rules.", 77))
        # A model the core does not serve: the core checks the settings after the answer (with its
        # input_tokens; core.yaml's unlisted_models: report), and the error is nonfatal.
        update = {"delegation": {"responses": {"model": "org/unlisted-model"}}}
        self.ws.send({"type": "session.update", "event_id": "u2", "session": update})
        error = (await self.ws.wait_for("error"))["error"]
        self.assertEqual(
            (error["code"], error["param"], error["client_event_id"]),
            ("responses_validation_failed", "session.delegation.responses.model", "u2"),
        )
        self.assertEqual(self.live_session.status, "active")

    async def test_the_benchmark_profile_starts_tau2s_session_with_no_error_and_runs_its_model(self):
        # The second real-core case of the background check: core.bench.yaml's unlisted_models: fallback runs a model
        # the catalog does not list on the slot's endpoint. So tau2's session, which names its own agent model, starts
        # with no error (core.yaml reports the model, as above), and its typed run goes to that endpoint as that model.
        await self.serve(
            TAU2_SESSION,
            [
                [chunk("Your order is on its way.", finish="stop")],  # the backend: the typed run's round
                _decision_chunks('{"action":"speak","speech":"Your order is on its way."}'),  # the writer
            ],
            core=CORE_BENCH_YAML,
        )
        await asyncio.gather(*self.live_session.delegation.checks)  # the start's settings check
        typed = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Where is my order?"}]}
        self.ws.send({"type": "response.item.create", "event_id": "q1", "item": typed})
        self.ws.send({"type": "response.create", "event_id": "go"})
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "way." in e["delta"])
        self.assertNotIn("error", self.ws.kinds())
        self.assertEqual(self.chat.requests[0]["model"], "tau2/agent-model")

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
        await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                [chunk("We have Earl Grey.", finish="stop")],
                _decision_chunks('{"action":"speak","speech":"We have Earl Grey."}'),
            ],
        )
        self.ws.send({"type": "session.thinking.append", "delegation_id": None, "content": "The caller is a regular."})
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Grey" in e["delta"])

        history = await self.history()
        self.assertEqual(history, json.loads(json.dumps(history)))
        frontend = [m["content"] for m in history["frontend"]["messages"]]
        self.assertEqual(history["frontend"]["model"], ENDPOINT.model)
        self.assertTrue(any("What teas do you have?" in text for text in frontend))
        self.assertTrue(any("The caller is a regular." in text for text in frontend))
        backend = history["backend"]
        request = backend["items"][0]["content"][0]["text"]
        # The delegation's input is the labeled transcript since the last request, the context included.
        self.assertTrue(request.startswith(TRANSCRIPT_HEADER), request)
        self.assertIn("The caller is a regular.", request)
        self.assertIn("What teas do you have?", request)
        self.assertIn("We have Earl Grey.", json.dumps(backend["items"][1:]))
        # The transcript went with the delegation. The lead-in was said after its input was built, so the next
        # delegation carries it; the answer the caller heard in full is the backend's own.
        self.assertEqual(backend["pending"], [f"{Heard.AGENT}: Let me check."])
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
        self.assertTrue(any("routing rules" in source for source in talker))
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

    async def test_history_names_a_session_update_as_the_source_of_the_backends_new_instructions(self):
        # The delegation side reads the session's stored settings, as session.update changed them. A
        # session that sets no instructions runs on the core's default, which the core reports on each response it
        # opens: the gateway knows it from the first request on.
        session = {**RESPONSES_SESSION, "delegation": {"type": "responses", "responses": {"model": "org/org/model"}}}
        await self.serve(
            session,
            [
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                [chunk("We have Earl Grey.", finish="stop")],
                _decision_chunks('{"action":"speak","speech":"We have Earl Grey."}'),
            ],
        )
        self.assertEqual((await self.history())["backend"]["prompt"], [])  # no request yet: the client set none
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Grey" in e["delta"])
        composed = compose(STATIC_BACKEND, *self.im().instruction_notes())
        self.assertEqual((await self.history())["backend"]["prompt"], [{"source": COMPOSED_SOURCE, "text": composed}])
        update = {"delegation": {"responses": {"instructions": "Use the tools."}}}
        self.ws.send({"type": "session.update", "event_id": "u", "session": update})
        await self.ws.wait_for("session.updated")
        self.assertEqual(
            (await self.history())["backend"]["prompt"][0],
            {"source": "delegation.responses.instructions", "text": "Use the tools."},
        )

    async def test_history_shows_the_voice_note_the_core_adds_v2_by_default(self):
        # The voice note is the core's (core.yaml delegation.voice_note, v2 by default); it follows the
        # session's instructions in what the core composes, which the newest response reports.
        await self.serve(
            RESPONSES_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                [chunk("We have Earl Grey.", finish="stop")],
                _decision_chunks('{"action":"speak","speech":"We have Earl Grey."}'),
            ],
        )
        await self.stt.say("What teas do you have?")
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "Grey" in e["delta"])
        composed = (await self.history())["backend"]["prompt"][-1]
        note = CoreSettings.load(CORE_YAML).texts.voice_notes["v2"]
        self.assertEqual(composed["source"], COMPOSED_SOURCE)
        self.assertTrue(composed["text"].startswith("Look things up.\n## Voice channel"), composed["text"])
        self.assertIn(note.strip(), composed["text"])

    async def test_history_shows_the_frame_prompt_set_v2_puts_around_the_instructions(self):
        # v2 frames the client's instructions: its role comes before them, how they apply after them.
        await self.serve(RESPONSES_SESSION, [], live_config=replace(LIVE_CONFIG, prompt_version="v2"))
        sources = [part["source"] for part in (await self.history())["frontend"]["prompt"]]
        self.assertEqual(sources[:2], ["prompt set v2: the voice's role", "session.instructions"])
        routing = sources.index("prompt set v2: routing rules")
        self.assertEqual(sources[routing - 1], "prompt set v2: how the instructions apply")

    async def test_history_shows_the_backend_model_a_session_update_chose(self):
        # The model the session's stored settings name, which the next request runs on, not the endpoint's: the
        # other /history tests run on the endpoint's own model, so they cannot tell the two apart.
        await self.serve(RESPONSES_SESSION, [])
        update = {"delegation": {"responses": {"model": "org/other-model"}}}
        self.ws.send({"type": "session.update", "event_id": "u", "session": update})
        await self.ws.wait_for("session.updated")
        self.assertEqual((await self.history())["backend"]["model"], "org/other-model")

    async def test_history_shows_the_backend_busy_while_a_delegation_waits_for_the_app(self):
        # The console's "working" flag comes from the adapter's work note.
        # At the core's client-tool deadline (0.2 s here, core.yaml client_tool_timeout_seconds) the core's work ends
        # and nothing runs there any more, but the delegation still waits for the app in the work note: when the
        # voice's app wait (0.4 s here) runs out, it says the app-wait line once and keeps the delegation watched, so a
        # late result is still said, and the flag stays on.
        live_config = replace(LIVE_CONFIG, delegation=replace(LIVE_CONFIG.delegation, app_wait_seconds=0.4))
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
        line = catalog_lines().app_wait_line
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: line.split()[-1] in e["delta"])
        self.assertTrue((await self.history())["backend"]["busy"])  # still watched after the app-wait line


class TypedQuestionTests(EngineHarness):
    """The application's typed question enters the talker's memory, with the gateway's half."""

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
        question = {"role": "user", "content": "Application context (reference data): Is there parking?"}
        # The writer's request: the question, then the answer to word, in its one user message.
        wording = self.chat.requests[1]["messages"][-1]
        self.assertEqual(wording["role"], "user")
        self.assertTrue(wording["content"].startswith(f"{question['content']}\n"), wording["content"])
        memory = self.engines[0].cascade.memory.messages()
        self.assertIn(question, memory)
        self.assertNotIn({"role": "user", "content": "Is there parking?"}, memory)  # no caller turn
        self.assertEqual([entry.kind.value for entry in self.engines[0].cascade.heard_log.entries], ["Agent"])


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
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "fact." in e["delta"], timeout=3)
        # The talker delegated with no lead-in, so the voice says its own first; the writer's answer may not
        # stay silent, so the client's update is said as written.
        self.assertEqual(" ".join(self.tts.spoken), f"{catalog_lines().lead_in} Background fact.")

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

        async def factory(config, transport, connection):
            built.append(config.mode)
            return SessionParts(frontend=_IdleFrontend())

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
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.post("/v1/live/sessions", json={"session": session, "transport": offer})
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(built, ["client"])
        live = next(iter(gateway.sessions.values()))
        self.assertEqual((live.delegation.target, live.delegation.conversation), ("client", None))
        await gateway.close_all()


class ConsoleClientLoopTests(EngineHarness):
    """The Live Console's client mode, played call by call against the real gateway and the real route.

    The test makes the console's requests in its order (``client/public/live/client-mode.js``): for each client
    delegation, a silent thinking append, then rounds of ``POST /v1/live/delegate``, then the answer as a commentary
    append. Only the backend model client is scripted. It leaves out the transcript's 1.5 s join window and 40-line cap,
    and the **Backend instructions** box.
    """

    TOOL_ROUNDS = 6  # client-mode.js: rounds that may run tools; one more round only answers

    CONSOLE_SESSION = {
        "model": "live-1",
        "audio": {"output": {"voice": "marin"}},
        "instructions": "You are a kiosk voice assistant.",
        "delegation": {"type": "client"},
    }

    async def asyncSetUp(self):
        import os

        import server
        from delegation_core.backend.providers import base as providers
        from tests.unit.frontend_backend_live.root.root_helpers import clear_root_caches
        from utils import clear_service_context, set_active_services

        self.backend = FakeChat(
            tool_call_round("call_1"),
            [chunk("We have green tea and chai."), chunk(finish="stop")],
            [chunk("We have espresso."), chunk(finish="stop")],
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        # The server's routes with the bundled files and the registry's hooks; the catalog's self-hosted section, named
        # outright, and no NVIDIA Cloud key, so no reachability probe picks another. With no token, the console's host
        # name must be allowed.
        live = {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "LIVE_ALLOWED_HOSTS": "console"}
        stack.enter_context(patch.dict(os.environ, {**live, "SERVICE_RECIPE": "server", "NVIDIA_API_KEY": ""}))
        for name in ("SERVICES_PATH", "LIVE_CONFIG_PATH", "CORE_CONFIG_PATH", "LIVE_ADAPTER_FACTORY"):
            os.environ.pop(name, None)  # the patch above restores the whole environment when the test ends
        # The route's rounds reach the root's one core, whose providers make their clients here: the scripted backend.
        # The session's own core is the harness's, which no backend request may reach in client mode.
        stack.enter_context(patch.object(providers, "AsyncOpenAI", return_value=self.backend))
        clear_root_caches(self)
        self.addCleanup(clear_service_context)
        self.addCleanup(set_active_services, None)
        routes = server.create_app()
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=routes), base_url="http://console")
        self.addAsyncCleanup(self.http.aclose)
        self.event_seq = 0

    def next_event_id(self, prefix):
        self.event_seq += 1
        return f"{prefix}_{self.event_seq:03d}"

    def transcript_context(self):
        """``Transcript.context``: the transcript deltas, joined by speaker, as the last eight "role: text" lines."""
        lines: list[list[str]] = []
        for event in self.ws.from_server:
            role = {"session.input_transcript.delta": "user", "session.output_transcript.delta": "assistant"}.get(
                event["type"]
            )
            if role is None:
                continue
            if lines and lines[-1][0] == role:
                lines[-1][1] += event["delta"]
            else:
                lines.append([role, event["delta"]])
        return "\n".join(f"{role}: {text.strip()}" for role, text in lines[-8:] if len(f"{role}: {text.strip()}") > 7)

    async def run_client_delegation(self, delegation_id, model, tools):
        """``runClientDelegation`` for one client delegation; returns the answer the console sends back.

        The delegation keeps its own history; rounds 1-6 may run tools, a seventh only answers; the answer is the
        rounds' ``final_answer`` messages (a message with no phase is one), and with none the console says a fixed line.
        """
        self.ws.send(
            {
                "type": "session.thinking.append",
                "event_id": self.next_event_id("progress"),
                "delegation_id": delegation_id,
                "content": "Working on that now. Nothing has been ordered yet.",
            }
        )
        reconstructed = (
            f"Recent voice transcript (may contain errors or corrections):\n{self.transcript_context()}\n\n"
            'Current cart state: {"lines":[],"item_count":0,"subtotal":0,"tax":0,"total":0,"currency":"USD"}\n\n'
            "The voice assistant delegated the user's latest request to you. Handle it and return "
            "a short result suitable for reading aloud."
        )
        history = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": reconstructed}]}]
        final, tool_ran = "", False
        for number in range(1, self.TOOL_ROUNDS + 2):
            body = {"input": history, "model": model, "tools": tools}
            response = await self.http.post("/v1/live/delegate", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            output = response.json()["response"]["output"]
            history.extend(output)
            calls = [item for item in output if item["type"] == "function_call"]
            if not calls or number > self.TOOL_ROUNDS:
                final = "\n\n".join(
                    "".join(part.get("text", "") for part in item["content"] if part.get("type") == "output_text")
                    for item in output
                    if item["type"] == "message" and item.get("phase") != "commentary"
                )
                break
            tool_ran = True
            history.extend(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": '{"items": ["green tea", "chai"]}',
                }
                for call in calls
            )
        if not final:  # client-mode.js PARTLY_DONE_LINE or FAILED_LINE
            final = "I may have only partly finished that" if tool_ran else "I could not get that done"
        self.ws.send(
            {
                "type": "session.commentary.append",
                "event_id": self.next_event_id("result"),
                "delegation_id": delegation_id,
                "content": final[:1200],
            }
        )
        return final

    async def test_the_console_runs_each_client_delegation_through_the_route_and_its_answer_is_spoken(self):
        info = (await self.http.get("/v1/live/info")).json()
        model, tools = info["backend"]["model"], info["sample_tools"]  # applyInfo and backendModel in app.js
        await self.serve(
            self.CONSOLE_SESSION,
            [
                _decision_chunks('{"action":"delegate","speech":"One moment."}'),
                _decision_chunks('{"action":"speak","speech":"We have green tea and chai."}'),
                _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
                _decision_chunks('{"action":"speak","speech":"We have espresso."}'),
            ],
        )
        # The first delegation: announced to the console, which runs it at once and answers with commentary that
        # names it.
        await self.stt.say("What teas do you have?")
        first = await self.ws.wait_for("session.delegation.created")
        self.assertEqual(first["delegation"]["target"], "client")
        self.assertNotIn("response_id", first["delegation"])
        answer = await self.run_client_delegation(first["delegation"]["id"], model, tools)
        self.assertEqual(answer, "We have green tea and chai.")
        # Round 1 sent the request with the cafe tools; round 2 resent the delegation's history, and the function's
        # output reached the backend model.
        self.assertEqual([request["model"] for request in self.backend.requests], [model, model])
        self.assertEqual(
            [tool["function"]["name"] for tool in self.backend.requests[0]["tools"]],
            [tool["name"] for tool in tools],
        )
        second_round = self.backend.requests[1]["messages"]
        self.assertEqual([m["role"] for m in second_round], ["system", "user", "assistant", "tool"])
        self.assertIn("corrections):\nuser: What teas do you have?", second_round[1]["content"])
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "chai" in e["delta"])
        writer = self.chat.requests[1]["messages"][-1]["content"]
        self.assertIn("We have green tea and chai.", writer)
        self.assertIn('you already told the caller: "One moment."', writer)
        self.feed_silence(10)
        await self.ws.wait_for("session.thinking.appended")
        await self.ws.wait_for("session.commentary.appended")
        # The history panel gets no server backend and shows the console's own history instead (refreshHistory).
        self.assertIsNone((await self.history())["backend"])
        # The second delegation's first round carries only its own rebuilt request: none of the first one's
        # items, and the backend model answers it.
        await self.stt.say("And coffee?")
        second = await self.ws.wait_for(
            "session.delegation.created", where=lambda e: e["delegation"]["id"] != first["delegation"]["id"]
        )
        self.assertEqual(
            await self.run_client_delegation(second["delegation"]["id"], model, tools), "We have espresso."
        )
        third_round = self.backend.requests[2]["messages"]
        self.assertEqual([m["role"] for m in third_round], ["system", "user"])
        self.assertIn("And coffee?", third_round[1]["content"])
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "espresso" in e["delta"])
        self.assertIn('you already told the caller: "Let me check."', self.chat.requests[3]["messages"][-1]["content"])
        # The session itself made only the talker's and the writer's requests: no backend request, no error, and no
        # response.event.
        self.assertEqual(len(self.chat.requests), 4)
        self.assertEqual(len(self.backend.requests), 3)
        self.assertNotIn("error", self.ws.kinds())
        self.assertNotIn("response.event", self.ws.kinds())


class _IdleFrontend:
    """A frontend adapter that starts and closes and does nothing else, for a session that only needs to start."""

    async def start(self, session, delegation):
        self.delegation = delegation

    async def close(self):
        pass


class CommentaryEndToEndTests(EngineHarness):
    """The client's commentary with no delegation id, in a responses session: the writer phrases it and it is spoken."""

    async def test_the_clients_commentary_is_spoken(self):
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"speak","speech":"It is sunny today."}')])
        self.ws.send(
            {
                "type": "session.commentary.append",
                "event_id": "c1",
                "delegation_id": None,
                "content": "Weather: sunny, 72 degrees.",
            }
        )
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "sunny" in e["delta"])
        # No talker turn comes first, so the writer's request is the first model request. The client's commentary is
        # worded with what else is ready as one line the caller hears, so the writer may only speak.
        update = self.chat.requests[0]["messages"][-1]["content"]
        self.assertIn("Weather: sunny, 72 degrees.", update)
        self.assertEqual(
            self.chat.requests[0]["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"],
            ["speak"],
        )

    async def test_client_commentary_is_spoken_even_when_the_writer_would_stay_silent(self):
        await self.serve(RESPONSES_SESSION, [_decision_chunks('{"action":"listen","speech":""}')])
        self.ws.send({"type": "session.commentary.append", "delegation_id": None, "content": "Background fact."})
        self.feed_silence(6)
        await self.ws.wait_for("session.commentary.appended", timeout=3)
        await self.ws.wait_for("session.output_transcript.delta", where=lambda e: "fact." in e["delta"], timeout=3)
        self.assertEqual(" ".join(self.tts.spoken), "Background fact.")


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
            return cascade_parts(config, transport, connection, chat=chat, stt=stt, tts=tts)

        core = chat_core(chat)
        self.addAsyncCleanup(core.aclose)
        app = FastAPI()
        gateway = mount_live(app, factory=factory, core=core_hook(core))
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


class StartOnceTests(EngineHarness):
    """Who starts what in a live session: each loop runs once.

    The delegation side's ``open`` opens the conversation, whose IM session starts its placer, and the adapter's
    ``start`` the silence timer. Attaching the cascade to its processor starts neither.
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

    The adapter's close stops the pipeline, the presenter's task, the silence timer and the talker's model, and leaves
    the delegation side alone. The close-order pin cannot tell, because the session closes the side right after the
    adapter, and a second close of the side does nothing.
    """

    async def test_the_adapters_close_leaves_the_delegation_side_to_its_own_close(self):
        await self.serve(RESPONSES_SESSION, [])
        # Closed here while the session is active, the adapter's pipeline end would ask the session to close, and the
        # session's own close would then close the side too; that request is held, so the adapter's close runs alone.
        with patch.object(self.engines[0].session, "request_close"):
            await self.parts[0].frontend.close()
        delegation = self.live_session.delegation
        placer = self.im().placer  # the IM session's loop in the core
        self.assertFalse(placer.done(), "the adapter's close stopped the delegation side's work")
        await delegation.close()
        self.assertTrue(placer.done())


class CallerAudioTests(EngineHarness):
    async def test_websocket_caller_audio_enters_the_pipeline_at_the_cascade_rate(self):
        # The cascade hears the caller at 16 kHz (LIVE_AUDIO_IN_RATE). Audio from a primary
        # WebSocket comes at the 24 kHz the gateway exchanges with the adapter, and the gate resamples it to the
        # cascade's rate before the pipeline sees it.
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


class ContextMeterTests(unittest.TestCase):
    def test_the_cascade_reports_how_much_conversation_its_talker_holds(self):
        from types import SimpleNamespace

        from tests.unit.cascade_helpers import ListConversation

        engine = SimpleNamespace(cascade=None)
        self.assertIsNone(CascadeAdapter.context_tokens(engine))
        engine.cascade = SimpleNamespace(conversation=ListConversation([{"role": "user", "content": "hi"}]))
        small = CascadeAdapter.context_tokens(engine)
        engine.cascade = SimpleNamespace(conversation=ListConversation([{"role": "user", "content": "hi " * 500}]))
        self.assertGreater(CascadeAdapter.context_tokens(engine), small + 300)


class AdapterCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_later_close_stops_only_a_pipeline_that_started_after_the_first(self):
        from types import SimpleNamespace

        calls = []

        async def stop_pipeline():
            calls.append("pipeline stopped")

        async def close_cascade():
            calls.append("cascade closed")

        adapter = SimpleNamespace(
            _closed=False, runner_task=None, stop_pipeline=stop_pipeline, cascade=SimpleNamespace(close=close_cascade)
        )
        # The session closes its frontend adapter, then its delegation side, and one adapter is both.
        await CascadeAdapter.close(adapter)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "cascade closed"])
        # A session closed while it was starting: its pipeline started after the close, and the start's early-close
        # path closes the parts again. That late pipeline is stopped; the cascade is not closed twice.
        adapter.runner_task = SimpleNamespace(done=lambda: False)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "cascade closed", "pipeline stopped"])
        adapter.runner_task = SimpleNamespace(done=lambda: True)
        await CascadeAdapter.close(adapter)
        self.assertEqual(calls, ["pipeline stopped", "cascade closed", "pipeline stopped"])


if __name__ == "__main__":
    unittest.main()
