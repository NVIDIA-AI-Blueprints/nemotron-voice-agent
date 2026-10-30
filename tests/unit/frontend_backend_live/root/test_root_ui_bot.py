# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The repository UI bot: greeting, display, playout, close."""

import asyncio
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from pipecat.runner.types import RunnerArguments

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.cascade.memory import TalkerMemory, split_messages
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.factory import adapter_block, ui_bot_core, with_display_label
from examples.frontend_backend_live.media import playout
from examples.frontend_backend_live.media.services import LineTextHold
from examples.frontend_backend_live.media.task_cards import RtviTaskCards
from examples.frontend_backend_live.prompts import get_prompt_set
from examples.frontend_backend_live.prompts.catalog import CHANNEL_PROMPT_KEY, catalog_prompt
from examples.shared.pipeline_utils import register_session_start_handlers
from realtime.lifecycle import announce_response, finish_response
from realtime.serializer import RealtimeFrameSerializer
from tests.unit.frontend_backend_live.root.root_helpers import CATALOG

PROMPTS = Path(pipeline.__file__).with_name("prompts.yaml")
INTRO = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["lines"]["intro"]  # the greeting the catalog holds
# What a Realtime client sends to type a line, and to ask for a response.
TYPED = {
    "type": "conversation.item.create",
    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A large latte."}]},
}
RESPONSE_CREATE = {"type": "response.create"}


class Recorder:
    """Handlers by event name, as Pipecat's ``event_handler`` decorator registers them."""

    def __init__(self):
        self.handlers = {}

    def event_handler(self, name):
        def register(handler):
            self.handlers[name] = handler
            return handler

        return register


class FakeTransport(Recorder):
    def input(self):
        return "transport in"

    def output(self):
        return "transport out"


class FakeWorker(Recorder):
    def __init__(self, processors, **kwargs):
        super().__init__()
        self.processors, self.kwargs, self.frames = processors, kwargs, []
        self.rtvi = Recorder()

    async def queue_frame(self, frame):
        self.frames.append(frame)

    async def cancel(self):
        pass


class FakeCascade:
    """The cascade as the bot builds it: its keywords, its greetings and its closes."""

    def __init__(self, case, agent, config, conversation, **kwargs):
        self.case, self.config, self.conversation, self.kwargs = case, config, conversation, kwargs
        self.greetings, self.closes = [], 0
        self.playout = SimpleNamespace()  # what the bot's playout tracker reports to
        case.cascades.append(self)

    def start(self):
        self.case.events.append("cascade started")

    def greet(self, text):
        self.greetings.append(text)

    async def close(self):
        self.closes += 1
        self.case.events.append("cascade closed")
        if self.case.close_fails:
            raise RuntimeError("the presenter would not stop")


class BotCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events, self.cascades, self.workers, self.started_with = [], [], [], {}
        self.aggregator_welcome, self.close_fails, self.during_run = [], False, None
        self.turn_ends, self.stt_turn_end = [], None
        self.transport, self.welcome = FakeTransport(), True
        self.body = {"pipeline_mode": "frontend-backend-live"}
        config = load_live_config(CONFIG_PATH)
        core = ui_bot_core(config)
        root = pipeline.BotRoot(LiveConfig.parse(with_display_label(adapter_block(config), core)), core, CATALOG)
        self.root, case = root, self

        class Delegation:
            def __init__(self, client):
                self.client = client

            async def open(self, view):
                case.events.append("delegation opened")

            async def close(self):
                case.events.append("delegation closed")

        class Core:
            async def aclose(self):
                case.events.append("core closed")

        class Runner:
            def __init__(self, handle_sigint=None):
                self.workers = []

            async def add_workers(self, worker):
                self.workers.append(worker)

            async def run(self):
                await case.during_run(self.workers[0])

        def worker(processors, **kwargs):
            built = FakeWorker(processors, **kwargs)
            case.workers.append(built)
            return built

        class TurnEnd:
            def __init__(self, speech):
                self.speech = speech
                case.turn_ends.append(self)

            def user_params(self, welcome):
                case.aggregator_welcome.append(welcome)
                return "params"

        def stt(body, entry, turn_end):
            case.stt_turn_end = turn_end
            return "stt"

        aggregator = Recorder()
        edges = {
            "bot_root": lambda: root,
            "create_transport": lambda runner_args: case.transport,
            "build_stt": stt,
            "build_tts": lambda body, entry: "tts",
            "LLMContextAggregatorPair": lambda context, user_params: SimpleNamespace(user=lambda: aggregator),
            "TURN_ENDS": {root.cascade.speech.turn_end: TurnEnd},  # only the turn end its speech settings name
            "build_core": lambda settings, *, endpoints, defaults: Core(),
            "core_client": lambda core: "client",
            "ServerDelegation": Delegation,
            "LiveTalkerProcessor": lambda: SimpleNamespace(speak="say", attach=lambda cascade: None),
            "create_audio_recorder": lambda: None,
            "Pipeline": lambda processors: processors,
            "UserBotLatencyObserver": Recorder,
            "with_realtime_observers": lambda *observers, transport: list(observers),
            "PipelineWorker": worker,
            "WorkerRunner": Runner,
            "register_session_start_handlers": lambda **kwargs: case.started_with.update(kwargs),
            "build_agent": lambda config, endpoint: "agent",
        }
        for name, value in edges.items():
            self.enterContext(patch.object(pipeline, name, value))
        self.enterContext(patch.object(pipeline.examples_registry, "welcome_message_enabled", lambda key: case.welcome))
        # The bot builds its cascade through the playout wiring it shares with the cascade adapter.
        self.enterContext(
            patch.object(playout, "CascadeSession", lambda *args, **kwargs: FakeCascade(case, *args, **kwargs))
        )

    async def run_bot(self, during_run):
        """Run ``bot()``; ``during_run(worker)`` plays what happens while its pipeline runs."""
        self.during_run = during_run
        await asyncio.wait_for(pipeline.bot(RunnerArguments(body=dict(self.body))), 5)
        (cascade,) = self.cascades
        return cascade

    async def client_ready(self, worker):
        """The client is ready: the shared start handler calls the bot's start hook."""
        await self.started_with["on_start"]()

    async def pipeline_finished(self, worker):
        await worker.handlers["on_pipeline_finished"](worker, "EndFrame")


class GreetingTests(BotCase):
    async def test_with_the_welcome_on_the_cascade_greets_with_its_own_line(self):
        cascade = await self.run_bot(self.client_ready)
        self.assertEqual(cascade.greetings, [INTRO])
        self.assertIs(self.started_with["welcome_enabled"], False)  # the shared handler adds no caller turn
        self.assertEqual(self.aggregator_welcome, [True])  # the caller's audio waits for the greeting
        roles = [message["role"] for message in cascade.conversation.messages()]
        self.assertNotIn("user", roles)  # the greeting is not a caller turn

    async def test_with_the_welcome_off_nothing_greets(self):
        self.welcome = False
        cascade = await self.run_bot(self.client_ready)
        self.assertEqual(cascade.greetings, [])
        self.assertIs(self.started_with["welcome_enabled"], False)
        self.assertEqual(self.aggregator_welcome, [False])


class TurnEndTests(BotCase):
    async def test_the_turn_end_its_speech_settings_name_builds_the_speech_to_text_and_the_turn_detection(self):
        await self.run_bot(self.client_ready)
        (turn_end,) = self.turn_ends
        self.assertIs(turn_end.speech, self.root.cascade.speech)
        self.assertIs(self.stt_turn_end, turn_end)  # the speech-to-text finalizes as the turn detection needs
        self.assertEqual(self.aggregator_welcome, [True])


class SeedTests(BotCase):
    """The talker's memory starts from the bot's seed: the talker's instructions, pinned, whatever the services set."""

    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ))
        for name in ("PROMPT_FILE_PATH", "PROMPT_SELECTOR"):  # the bot's talker is the catalog's own default
            os.environ.pop(name, None)

    async def seed(self) -> list[dict]:
        cascade = await self.run_bot(self.client_ready)
        return cascade.conversation.messages()

    async def test_the_bot_keeps_the_cafe_talker_and_its_channel_as_one_system_message(self):
        talker = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["talker"]["content"]
        channel = catalog_prompt(CHANNEL_PROMPT_KEY)
        self.assertEqual(await self.seed(), [{"role": "system", "content": f"{talker}\n\n{channel}"}])

    async def test_with_a_system_prompt_the_talkers_prompt_stays_through_compaction(self):
        self.body["system_prompt"] = "detailed thinking off"  # a services entry's control directive
        seed = await self.seed()
        self.assertEqual([message["role"] for message in seed], ["system", "system"])
        self.assertEqual(seed[0]["content"], "detailed thinking off")
        memory = TalkerMemory.seeded(get_prompt_set(""), seed, context_tokens=2000, compact_at=0.8)
        turns = 40
        for turn in range(turns):
            memory.caller(f"Turn {turn}: " + "a large latte with oat milk, please. " * 4)
        instructions, history = split_messages(memory.messages())
        self.assertEqual(instructions, "\n\n".join(message["content"] for message in seed))
        self.assertLess(len(history), turns)  # compaction ran: the oldest caller turns left, the instructions stayed
        self.assertTrue(all(message["content"].startswith("Turn ") for message in history))  # no prompt as a turn


class RealtimeGreetingTests(BotCase):
    """On ``/v1/realtime``, a client's typed text waits for the greeting, as the caller's audio does."""

    def setUp(self):
        super().setUp()
        self.sent = []
        self.serializer = RealtimeFrameSerializer()
        self.serializer.set_emit(self.emit)
        self.transport._realtime_serializer = self.serializer  # as the Realtime transport carries it
        self.body["protocol"] = "realtime"

        def start_handlers(**kwargs):
            self.started_with.update(kwargs)
            register_session_start_handlers(**kwargs)  # the shared handler itself

        self.enterContext(patch.object(pipeline, "register_session_start_handlers", start_handlers))

    async def emit(self, event):
        self.sent.append(event)

    async def client_connected(self):
        """The Realtime client connects: the shared start handler calls the bot's start hook."""
        await self.transport.handlers["on_client_connected"](self.transport, "client")

    async def sends(self, message):
        """The client sends ``message``: the frame the serializer hands the pipeline, or None when it refuses it."""
        return await self.serializer.deserialize(json.dumps(message))

    async def greeting_done(self):
        """The greeting's response is done, as the Realtime observer finishes it once its audio has played."""
        await announce_response(self.serializer.conversation, self.emit)
        await finish_response(self.serializer.conversation, self.emit, status="completed")

    async def test_with_the_welcome_on_typed_text_waits_until_the_greeting_has_been_said(self):
        async def during_run(worker):
            await self.client_connected()
            self.assertIsNone(await self.sends(TYPED))
            self.assertIsNone(await self.sends(RESPONSE_CREATE))
            refused = [event["error"]["code"] for event in self.sent if event["type"] == "error"]
            self.assertEqual(refused, ["item_rejected_pre_intro", "response_create_rejected_pre_intro"])
            await self.greeting_done()
            self.assertIsNotNone(await self.sends(TYPED))

        cascade = await self.run_bot(during_run)
        self.assertEqual(cascade.greetings, [INTRO])

    async def test_with_the_welcome_off_typed_text_is_taken_at_once(self):
        self.welcome = False

        async def during_run(worker):
            await self.client_connected()
            self.assertIsNotNone(await self.sends(TYPED))

        cascade = await self.run_bot(during_run)
        self.assertEqual(cascade.greetings, [])


class DisplayTests(BotCase):
    async def test_each_delegation_is_a_task_card_on_the_bots_worker(self):
        cascade = await self.run_bot(self.client_ready)
        display = cascade.kwargs["display"]
        self.assertIsInstance(display, RtviTaskCards)
        self.assertEqual(display.queue_frame, self.workers[0].queue_frame)


class PlayoutWiringTests(BotCase):
    async def test_a_stitched_text_to_speech_holds_each_lines_text_and_the_tracker_follows_the_output(self):
        tts = SimpleNamespace(settings=SimpleNamespace(synthesis_mode="stitched"))  # Magpie's own mode
        self.enterContext(patch.object(pipeline, "build_tts", lambda body, entry: tts))
        cascade = await self.run_bot(self.client_ready)
        processors = self.workers[0].processors
        speech_out = processors[processors.index(tts) :]
        self.assertEqual(len(speech_out), 4, speech_out)
        self.assertIsInstance(speech_out[1], LineTextHold)  # a stitched line's text waits for its audio
        self.assertEqual(speech_out[2], "transport out")
        self.assertIsInstance(speech_out[3], playout.PlayoutTracker)  # what really played, read after the output
        reports = {key: cascade.kwargs.get(key) for key in ("reports_playout", "reports_sentences")}
        self.assertEqual(reports, {"reports_playout": True, "reports_sentences": False})


class CloseTests(BotCase):
    async def test_the_cascade_closes_when_the_pipeline_finishes_and_only_once(self):
        async def during_run(worker):
            await self.pipeline_finished(worker)
            self.events.append("pipeline ended")

        cascade = await self.run_bot(during_run)
        self.assertEqual(cascade.closes, 1)
        self.assertEqual(
            self.events,
            [
                "delegation opened",
                "cascade started",
                "cascade closed",
                "pipeline ended",
                "delegation closed",
                "core closed",
            ],
        )

    async def test_the_cascade_closes_when_the_bot_ends_without_a_finished_pipeline(self):
        async def during_run(worker):
            self.events.append("pipeline ended")

        cascade = await self.run_bot(during_run)
        self.assertEqual(cascade.closes, 1)
        self.assertEqual(self.events[-4:], ["pipeline ended", "cascade closed", "delegation closed", "core closed"])

    async def test_a_cascade_that_fails_to_close_still_lets_the_delegation_and_the_core_close(self):
        self.close_fails = True
        cascade = await self.run_bot(self.pipeline_finished)
        self.assertEqual(cascade.closes, 1)
        self.assertEqual(self.events[-2:], ["delegation closed", "core closed"])


if __name__ == "__main__":
    unittest.main()
