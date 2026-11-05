# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The repository UI bot as it runs: greeting, seed, Realtime clients, close."""

import asyncio
import json
import os
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import yaml
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation_core.profiles import load_profile
from examples.frontend_backend_live.factory import adapter_block, ui_bot_core, with_display_label
from examples.frontend_backend_live.frontend import build_frontend
from examples.frontend_backend_live.talker import TalkerChat
from gateways.realtime.lifecycle import announce_response, finish_response
from gateways.realtime.serializer import RealtimeFrameSerializer
from tests.unit.cascade_helpers import FakeSTT, transcript_turns
from tests.unit.frontend_backend_live.root.root_helpers import CATALOG
from tests.unit.frontend_helpers import Acts, Port, ScriptedChat, SentenceTTS, SpeakingOutput, decision

# The example's prompts.yaml as written.
PROMPTS_FILE = yaml.safe_load(CONFIG_PATH.with_name("prompts.yaml").read_text(encoding="utf-8"))


class RecordingOutput(SpeakingOutput):
    """The transport's output: speaks as :class:`SpeakingOutput` does and keeps each RTVI message the bot sends."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: list[dict] = []

    async def process_frame(self, frame, direction):
        if isinstance(frame, RTVIServerMessageFrame):
            self.messages.append(frame.data)
        await super().process_frame(frame, direction)


class FakeTransport:
    """The bot's RTVI transport: where caller audio enters and reply audio leaves, and its event handlers by name."""

    def __init__(self):
        self.handlers = {}
        self.inbound, self.outbound = Acts(name="transport in"), RecordingOutput(name="transport out")

    def input(self):
        return self.inbound

    def output(self):
        return self.outbound

    def event_handler(self, name):
        return lambda handler: self.handlers.setdefault(name, handler)


class BotCase(unittest.IsolatedAsyncioTestCase):
    """``pipeline.bot()`` as it runs: its own pipeline, on a scripted talker, fake speech and a fake RTVI transport.

    The bot's server delegation and core are stand-ins that record their opens and closes; a test that needs the
    bot's own core patches ``build_core``, ``core_client`` and ``ServerDelegation`` back.
    """

    def setUp(self):
        self.events, self.aggregator_welcome, self.welcome, self.close_fails = [], [], True, False
        self.chat, self.body = ScriptedChat(), {"pipeline_mode": "frontend-backend-live"}
        self.frontends, self.running = [], asyncio.Event()
        self.transport, self.stt, self.tts = FakeTransport(), FakeSTT(), SentenceTTS()
        config = load_live_config(load_profile, CONFIG_PATH)
        core = ui_bot_core(config)
        cascade = LiveConfig.parse(with_display_label(adapter_block(config), core))
        case = self

        class ScriptedTalker(TalkerChat):
            def create_client(self, api_key=None, base_url=None, **kwargs):
                return case.chat

        class Delegation(Port):
            def __init__(self, client):
                super().__init__()

            async def open(self, view):
                case.events.append("delegation opened")

            async def close(self):
                case.events.append("delegation closed")

        class Core:
            async def aclose(self):
                case.events.append("core closed")

        class TurnEnd:
            """The turn end the speech settings name: the whole-call tests' turn detection, with the bot's welcome."""

            def __init__(self, speech):
                pass

            def user_params(self, welcome):
                case.aggregator_welcome.append(welcome)
                return replace(transcript_turns(), user_turn_stop_timeout=0.3)

        def build(spec, edges):
            built = build_frontend(spec, edges)
            case.frontends.append(built)
            start, close = built.voice.start, built.voice.close

            def started():
                case.events.append("voice started")
                start()

            async def closed():
                case.events.append("voice closed")
                await close()
                if case.close_fails:
                    raise RuntimeError("the voice would not stop")

            built.voice.start, built.voice.close = started, closed
            built.worker.add_event_handler("on_pipeline_started", lambda worker, frame: case.running.set())
            return built

        root = pipeline.BotRoot(cascade, ScriptedTalker, core, config.ui_bot.channel, CATALOG)
        for name, value in {
            "bot_root": lambda: root,
            "create_transport": AsyncMock(side_effect=lambda runner_args: case.transport),
            "build_stt": lambda body, entry, turn_end: case.stt,
            "build_tts": lambda body, entry: case.tts,
            "TURN_ENDS": {cascade.speech.turn_end: TurnEnd},
            "build_core": lambda settings, *, endpoints, defaults: Core(),
            "core_client": lambda core: "client",
            "ServerDelegation": Delegation,
            "create_audio_recorder": lambda: None,
            "build_frontend": build,
        }.items():
            self.enterContext(patch.object(pipeline, name, value))
        self.enterContext(patch.object(pipeline.examples_registry, "welcome_message_enabled", lambda key: case.welcome))

    async def run_bot(self, during_run=None):
        """Run ``bot()``: ``during_run(frontend)`` plays the call once the pipeline runs, then the client leaves."""
        self.bot_task = asyncio.create_task(pipeline.bot(RunnerArguments(body=dict(self.body))))
        running = asyncio.create_task(self.running.wait())
        done, _ = await asyncio.wait({self.bot_task, running}, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        running.cancel()
        if self.bot_task in done:
            await self.bot_task  # raises what stopped the bot before its pipeline ran
        (frontend,) = self.frontends
        try:
            if during_run is not None:
                await during_run(frontend)
        finally:
            if not self.bot_task.done():
                await self.transport.handlers["on_client_disconnected"](self.transport, "client")
            async with asyncio.timeout(10):
                await asyncio.gather(self.bot_task, return_exceptions=True)
        return frontend

    async def until(self, predicate, what: str, timeout: float = 4.0) -> None:
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None


class GreetingTests(BotCase):
    async def test_with_the_welcome_on_the_voice_words_its_greeting_from_the_intro_line_on_the_cafe_seed(self):
        self.chat.script.append(decision("speak", "Welcome to Bluebird."))
        self.body["system_prompt"] = "detailed thinking off"  # a services entry's control directive comes first
        self.enterContext(patch.dict(os.environ))
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):  # the bot's talker is the registry default's
            os.environ.pop(name, None)

        async def during_run(frontend):
            await frontend.worker.rtvi.set_client_ready()  # the shared start handler calls the bot's start hook
            await self.until(lambda: self.tts.spoken, "the greeting")

        frontend = await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, ["Welcome to Bluebird."])
        intro = PROMPTS_FILE["agent_prompts"]["Lines"]["intro"]["content"]
        self.assertIn(intro, json.dumps(self.chat.requests[0]["messages"], ensure_ascii=False))
        talker = PROMPTS_FILE["cafe"]["content"]
        channel = PROMPTS_FILE["agent_prompts"]["Channel"]["cafe"]["content"].strip()
        self.assertIn(
            f"detailed thinking off\n\n{talker}\n\n{channel}", self.chat.requests[0]["messages"][0]["content"]
        )
        self.assertEqual(self.aggregator_welcome, [True])  # the caller's audio waits for the greeting
        self.assertNotIn("caller", [message["kind"] for message in frontend.voice.conversation()])

    async def test_with_the_welcome_off_nothing_greets(self):
        self.welcome = False

        async def during_run(frontend):
            await frontend.worker.rtvi.set_client_ready()
            await asyncio.sleep(0.2)

        await self.run_bot(during_run)
        self.assertEqual((self.tts.spoken, self.chat.requests, self.aggregator_welcome), ([], [], [False]))


class RealtimeGreetingTests(BotCase):
    """On ``/v1/realtime``, a client's typed text waits for the greeting, as the caller's audio does."""

    TYPED = {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A large latte."}]},
    }

    def setUp(self):
        super().setUp()
        self.sent, self.serializer = [], RealtimeFrameSerializer()
        self.serializer.set_emit(self.emit)
        self.transport._realtime_serializer = self.serializer  # as the Realtime transport carries it
        self.body["protocol"] = "realtime"
        self.chat.script.append(decision("speak", "Welcome to Bluebird."))

    async def emit(self, event):
        self.sent.append(event)

    async def sends(self, message):
        """The client sends ``message``: the frame the serializer hands the pipeline, or None when it refuses it."""
        return await self.serializer.deserialize(json.dumps(message))

    async def test_with_the_welcome_on_typed_text_waits_until_the_greeting_has_been_said(self):
        async def during_run(frontend):
            await self.transport.handlers["on_client_connected"](self.transport, "client")
            self.assertIsNone(await self.sends(self.TYPED))
            self.assertIsNone(await self.sends({"type": "response.create"}))
            refused = [event["error"]["code"] for event in self.sent if event["type"] == "error"]
            self.assertEqual(refused, ["item_rejected_pre_intro", "response_create_rejected_pre_intro"])
            await self.until(lambda: self.tts.spoken, "the greeting")
            # The greeting's response is done, as the Realtime observer finishes it once its audio has played.
            await announce_response(self.serializer.conversation, self.emit)
            await finish_response(self.serializer.conversation, self.emit, status="completed")
            self.assertIsNotNone(await self.sends(self.TYPED))

        await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, ["Welcome to Bluebird."])

    async def test_with_the_welcome_off_typed_text_is_taken_at_once(self):
        self.welcome = False

        async def during_run(frontend):
            await self.transport.handlers["on_client_connected"](self.transport, "client")
            self.assertIsNotNone(await self.sends(self.TYPED))

        await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, [])


class CloseTests(BotCase):
    async def test_the_voice_the_delegation_and_the_core_close_once_in_order_however_the_bot_ends(self):
        for how in ("finished", "cancelled", "the voice fails to close"):
            with self.subTest(how):
                self.events.clear()
                self.frontends, self.running, self.transport = [], asyncio.Event(), FakeTransport()
                self.close_fails = how == "the voice fails to close"

                async def during_run(frontend, how=how):
                    if how == "cancelled":
                        self.bot_task.cancel()
                    else:
                        await frontend.worker.cancel()

                await self.run_bot(during_run)
                self.assertEqual(self.events.count("voice closed"), 1)
                self.assertEqual(self.events[-3:], ["voice closed", "delegation closed", "core closed"])
                if how == "finished":
                    self.assertEqual(self.events[:2], ["delegation opened", "voice started"])
