# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""The repository UI bot: greeting, display, close."""

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from pipecat.runner.types import RunnerArguments

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.adapter.task_cards import RtviTaskCards
from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.factory import adapter_block, ui_bot_core, with_display_label
from tests.unit.frontend_backend_live.root.root_helpers import CATALOG

PROMPTS = Path(pipeline.__file__).with_name("prompts.yaml")
INTRO = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))["lines"]["intro"]  # the greeting the catalog holds


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
        self.transport, self.welcome = FakeTransport(), True
        config = load_live_config(CONFIG_PATH)
        core = ui_bot_core(config)
        root = pipeline.BotRoot(LiveConfig.parse(with_display_label(adapter_block(config), core)), core, CATALOG)
        case = self

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

        def aggregator_params(welcome):
            case.aggregator_welcome.append(welcome)
            return "params"

        aggregator = Recorder()
        edges = {
            "bot_root": lambda: root,
            "create_transport": lambda runner_args: case.transport,
            "build_stt": lambda body, entry: "stt",
            "build_tts": lambda body, entry: "tts",
            "LLMContextAggregatorPair": lambda context, user_params: SimpleNamespace(user=lambda: aggregator),
            "build_user_aggregator_params": aggregator_params,
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
            "CascadeSession": lambda *args, **kwargs: FakeCascade(case, *args, **kwargs),
            "build_agent": lambda config, endpoint: "agent",
        }
        for name, value in edges.items():
            self.enterContext(patch.object(pipeline, name, value))
        self.enterContext(patch.object(pipeline.examples_registry, "welcome_message_enabled", lambda key: case.welcome))

    async def run_bot(self, during_run):
        """Run ``bot()``; ``during_run(worker)`` plays what happens while its pipeline runs."""
        self.during_run = during_run
        await asyncio.wait_for(pipeline.bot(RunnerArguments(body={"pipeline_mode": "frontend-backend-live"})), 5)
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


class DisplayTests(BotCase):
    async def test_each_delegation_is_a_task_card_on_the_bots_worker(self):
        cascade = await self.run_bot(self.client_ready)
        display = cascade.kwargs["display"]
        self.assertIsInstance(display, RtviTaskCards)
        self.assertEqual(display.queue_frame, self.workers[0].queue_frame)


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
