# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The repository UI bot as its root (``pipeline.py``) runs it: greeting, Realtime clients, whole calls, close."""

import asyncio
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml
from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.processors.filters.identity_filter import IdentityFilter
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments

from examples.frontend_backend_live import factory, pipeline
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.frontend import build_frontend
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.talker import TalkerChat, TalkerTools
from gateways.live.server_delegation import ServerDelegation
from gateways.realtime.lifecycle import announce_response, finish_response
from gateways.realtime.serializer import RealtimeFrameSerializer
from tests.unit.cascade_helpers import FakeSTT, chat_adapters, transcript_turns
from tests.unit.core_helpers import chunk, tool_fragment
from tests.unit.frontend_helpers import (
    CATALOG,
    FRONT,
    Port,
    ScriptedChat,
    SentenceTTS,
    SpeakingOutput,
    StubAdapter,
    decision,
    wait_until,
)

PROMPTS_FILE = yaml.safe_load(Path(pipeline.__file__).with_name("prompts.yaml").read_text(encoding="utf-8"))


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
        self.inbound, self.outbound = IdentityFilter(name="transport in"), RecordingOutput(name="transport out")

    def input(self):
        return self.inbound

    def output(self):
        return self.outbound

    def event_handler(self, name):
        return lambda handler: self.handlers.setdefault(name, handler)


class BotCase(unittest.IsolatedAsyncioTestCase):
    """``pipeline.bot()`` as it runs, on a scripted talker, fake speech and a fake RTVI transport."""

    def setUp(self):
        self.events, self.aggregator_welcome, self.welcome, self.close_fails = [], [], True, False
        self.chat, self.body = ScriptedChat(), {"pipeline_mode": "frontend-backend-live"}
        self.frontends, self.running = [], asyncio.Event()
        self.transport, self.stt, self.tts = FakeTransport(), FakeSTT(), SentenceTTS()
        config = factory.example_config()
        core = factory.ui_bot_core(config)
        cascade = LiveConfig.model_validate(factory.with_display_label(factory.adapter_block(config), core))
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

        async def core_closed():
            case.events.append("core closed")

        class TurnEnd:
            """Stand-in for the turn settings: the whole-call tests' turn detection, noting the welcome."""

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

        root = pipeline.BotRoot(cascade, core, config.ui_bot.channel, CATALOG)
        self.enterContext(patch.dict(pipeline.LLM_SERVICES, {"json": ScriptedTalker}))
        for name, value in {
            "bot_root": lambda: root,
            "create_transport": AsyncMock(side_effect=lambda runner_args: case.transport),
            "build_stt": lambda body, entry: case.stt,
            "build_tts": lambda body, entry: case.tts,
            "SmartTurn": TurnEnd,
            "build_core": lambda settings, *, endpoints, defaults: SimpleNamespace(aclose=core_closed),
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
            await self.bot_task
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

    until = staticmethod(wait_until)


class GreetingTests(BotCase):
    async def test_the_welcome_is_worded_from_the_intro_on_the_cafe_seed(self):
        self.chat.script.append(decision("speak", "Welcome to Bluebird."))
        self.body["system_prompt"] = "detailed thinking off"
        self.enterContext(patch.dict(os.environ))
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):
            os.environ.pop(name, None)

        async def during_run(frontend):
            await frontend.worker.rtvi.set_client_ready()
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
        self.assertEqual(self.aggregator_welcome, [True])
        self.assertNotIn("caller", [message["kind"] for message in frontend.voice.conversation()])

    async def test_with_the_welcome_off_nothing_greets(self):
        self.welcome = False

        async def during_run(frontend):
            await frontend.worker.rtvi.set_client_ready()
            await asyncio.sleep(0.2)

        frontend = await self.run_bot(during_run)
        self.assertEqual((self.tts.spoken, self.chat.requests, self.aggregator_welcome), ([], [], [False]))
        talker = frontend.voice.talker
        self.assertTrue(isinstance(talker, TalkerChat) and not isinstance(talker, TalkerTools))
        self.assertEqual(frontend.voice.routing_format.name, "json_schema")

    async def test_the_selected_model_gives_the_talker_style_and_level(self):
        case = self

        class ScriptedTools(TalkerTools):
            def create_client(self, api_key=None, base_url=None, **kwargs):
                return case.chat

        self.enterContext(patch.dict(pipeline.LLM_SERVICES, {"tools": ScriptedTools}))
        state = pipeline.bot_root()
        first = {**FRONT, "id": "local:first", "key": "first"}
        second = {**FRONT, "id": "local:second", "key": "second", "response_format": "none"}
        cascade = state.cascade.model_copy(update={"talker_style": {"second": "tools"}})
        root = replace(state, cascade=cascade, catalog=ExampleCatalog({**CATALOG.offered, "llm": [first, second]}))
        self.enterContext(patch.object(pipeline, "bot_root", lambda: root))
        self.body["llm_id"] = "local:second"
        frontend = await self.run_bot()
        self.assertIs(type(frontend.voice.talker), ScriptedTools)
        self.assertEqual(frontend.voice.routing_format.name, "tools")
        self.assertEqual(frontend.voice.talker.response_format.name, "none")


class RealtimeGreetingTests(BotCase):
    TYPED = {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A large latte."}]},
    }

    def setUp(self):
        super().setUp()
        self.sent, self.serializer = [], RealtimeFrameSerializer()
        self.serializer.set_emit(self.emit)
        self.transport._realtime_serializer = self.serializer
        self.body["protocol"] = "realtime"
        self.chat.script.append(decision("speak", "Welcome to Bluebird."))

    async def emit(self, event):
        self.sent.append(event)

    async def sends(self, message):
        """Send ``message`` as the client; return the frame the serializer hands the pipeline, or None if refused."""
        return await self.serializer.deserialize(json.dumps(message))

    async def test_typed_text_waits_for_the_greeting_when_it_is_on(self):
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

    async def test_typed_text_is_taken_at_once_when_the_welcome_is_off(self):
        self.welcome = False

        async def during_run(frontend):
            await self.transport.handlers["on_client_connected"](self.transport, "client")
            self.assertIsNotNone(await self.sends(self.TYPED))

        await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, [])


class CloseTests(BotCase):
    async def test_the_voice_delegation_and_core_close_once_in_order(self):
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


class BotCallTests(BotCase):
    def setUp(self):
        super().setUp()
        self.welcome, self.backend = False, ScriptedChat()

        def core(settings, *, endpoints, defaults):
            return build_core(settings, endpoints=endpoints, defaults=defaults, adapters=chat_adapters(self.backend))

        for name, value in {
            "build_core": core,
            "core_client": factory.core_client,
            "ServerDelegation": ServerDelegation,
        }.items():
            self.enterContext(patch.object(pipeline, name, value))

    def cards(self) -> dict:
        """Return each delegation's newest task card, by its task id."""
        messages = self.transport.outbound.messages
        return {data["task_id"]: data for data in messages if data.get("type") == "agent-task-update"}

    async def test_each_cafe_order_runs_the_cafes_tool_and_shows_its_card(self):
        first, second = asyncio.Event(), asyncio.Event()
        worded = "One large latte is in your cart, five ninety-seven with tax."
        self.chat.script += [
            decision("delegate", "Let me add that."),
            decision("speak", worded),
            decision("delegate", "One moment."),
            decision("speak", "Your Matcha is ordered."),
        ]
        arguments = '{"action":"add","item_id":"lat","size":"large"}'
        self.backend.script += [
            [chunk(tool_calls=[tool_fragment(0, "call_1", "manage_cart", arguments)]), chunk(finish="tool_calls")],
            first,
            [chunk("Added one large Latte. Your total is $5.97 with tax.", finish="stop")],
            second,
            [chunk("Ordered one Matcha.", finish="stop")],
        ]

        def heard(frontend) -> list[str]:
            return [m["content"] for m in frontend.context.get_messages() if m["role"] == "assistant"]

        async def during_run(frontend):
            await self.stt.propose("A large latte, please.")
            await self.until(lambda: self.tts.spoken, "the first lead-in")
            first.set()
            await self.until(lambda: len(heard(frontend)) == 2, "the first answer heard")
            await self.stt.propose("One Matcha, please.")
            await self.until(lambda: len(self.tts.spoken) == 3, "the second lead-in")
            second.set()
            await self.until(lambda: len(self.tts.spoken) == 4 and len(self.cards()) == 2, "the second answer")

        frontend = await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, ["Let me add that.", worded, "One moment.", "Your Matcha is ordered."])
        tool_result = next(m for m in self.backend.requests[1]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_result["content"])["status"], "added")
        relay = self.chat.requests[1]["messages"][-1]["content"]
        self.assertIn("Added one large Latte. Your total is $5.97 with tax.", relay)
        self.assertEqual(heard(frontend), self.tts.spoken)
        self.assertEqual(
            [(card["query"], card["status"], card["response"]) for card in self.cards().values()],
            [
                ("A large latte, please.", "done", "Added one large Latte. Your total is $5.97 with tax."),
                ("One Matcha, please.", "done", "Ordered one Matcha."),
            ],
        )

    async def test_a_finished_turn_and_its_latency_reach_the_client(self):
        self.chat.script.append(decision("speak", "Earl Grey it is."))

        async def during_run(frontend):
            await self.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))
            await self.stt.propose("A pot of Earl Grey.")
            await self.until(lambda: len(self.transport.outbound.messages) >= 4, "the turn's messages")

        await self.run_bot(during_run)
        sent = self.transport.outbound.messages
        kinds = ["user-turn-finalized", "user-bot-latency", "metric-group", "latency-breakdown"]
        self.assertEqual([data["type"] for data in sent], kinds)
        self.assertEqual((sent[0]["transcript"], sent[2]["group_id"]), ("A pot of Earl Grey.", "turn-1"))


class BotRootTests(unittest.TestCase):
    def test_the_ui_bot_reads_the_cascade_block_with_the_label(self):
        pipeline.bot_root.cache_clear()
        self.addCleanup(pipeline.bot_root.cache_clear)
        with (
            patch.object(pipeline.ExampleCatalog, "read", return_value=CATALOG),
            patch.object(pipeline, "frontend_adapters", return_value={"cascade": StubAdapter}),
        ):
            state = pipeline.bot_root()
        self.assertEqual(state.cascade.slot, "llm")
        self.assertEqual(state.cascade.display_label, state.core.delegation.display_label)
