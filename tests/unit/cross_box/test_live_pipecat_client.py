# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Pipecat's stock ``OpenAILiveLLMService`` client against the live gateway, over a real WebSocket."""

import asyncio
import unittest

import uvicorn
from fastapi import FastAPI
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndFrame, Frame, LLMContextFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.openai.live.llm import OpenAILiveLLMService
from pipecat.services.openai.responses.llm import OpenAIResponsesLLMService
from pipecat.workers.runner import WorkerRunner

from live.mount import mount_live
from tests.unit.cascade_helpers import (
    FakeSTT,
    FakeTTS,
    _decision_chunks,
    cascade_parts,
    chat_core,
    core_hook,
    tool_call_round,
)
from tests.unit.core_helpers import FakeChat, chunk
from tests.unit.live_helpers import FakeAdapter

INSTRUCTIONS = "Be brief."


class Tap(FrameProcessor):
    """Records the text frames the service pushes downstream."""

    def __init__(self):
        super().__init__()
        self.text: list[str] = []

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.text.append(frame.text)
        await self.push_frame(frame, direction)


class StockClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engines: list[FakeAdapter] = []

        async def factory(config, transport, connection):
            self.engines.append(FakeAdapter())
            return self.engines[-1].parts()

        app = FastAPI()
        self.gateway = mount_live(app, factory=factory, core=FakeAdapter().core)  # the fake is also the one core
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.serving = asyncio.create_task(self.server.serve())
        async with asyncio.timeout(10):
            while not self.server.started:
                await asyncio.sleep(0.02)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.should_exit = True
        async with asyncio.timeout(10):
            await self.serving

    async def run_client(self, delegation, scenario):
        """Run the stock service in a pipeline; ``scenario(engine, service)`` drives the server side."""
        service = OpenAILiveLLMService(
            api_key="test-token",
            base_url=f"ws://127.0.0.1:{self.port}/v1/live/sessions",
            settings=OpenAILiveLLMService.Settings(model="live-1", system_instruction=INSTRUCTIONS),
            delegation=delegation,
        )
        tap = Tap()
        started = asyncio.Event()
        service.add_event_handler("on_session_started", lambda *_: started.set())
        worker = PipelineWorker(Pipeline([service, tap]), enable_rtvi=False, idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frame(LLMContextFrame(LLMContext([{"role": "system", "content": INSTRUCTIONS}])))
            async with asyncio.timeout(10):
                await started.wait()
            await scenario(self.engines[0], tap)
            await worker.queue_frame(EndFrame())
            async with asyncio.timeout(15):
                await running
        finally:
            if not running.done():
                await worker.cancel()
                await asyncio.gather(running, return_exceptions=True)

    async def test_a_stock_client_starts_a_session_without_a_backend(self):
        async def scenario(engine, tap):
            self.assertTrue(engine.started)
            self.assertEqual(self.gateway.sessions[engine.session.id].config.mode, "client")

        await self.run_client(None, scenario)
        self.assertTrue(self.engines[0].closed)
        self.assertEqual(self.gateway.sessions, {})

    async def test_a_stock_client_starts_a_session_with_responses_delegation(self):
        async def scenario(engine, tap):
            self.assertEqual(self.gateway.sessions[engine.session.id].config.mode, "responses")
            self.assertEqual(self.gateway.sessions[engine.session.id].config.instructions, INSTRUCTIONS)

        delegation = OpenAILiveLLMService.ResponsesDelegation(
            settings=OpenAIResponsesLLMService.Settings(model="org/org/model")
        )
        await self.run_client(delegation, scenario)
        self.assertTrue(self.engines[0].closed)

    async def test_assistant_transcript_from_the_server_reaches_the_clients_pipeline(self):
        async def scenario(engine, tap):
            await engine.session.emit("session.output_transcript.delta", delta="Hello there. ", start_ms=0, end_ms=500)
            async with asyncio.timeout(5):
                while not tap.text:
                    await asyncio.sleep(0.02)
            self.assertIn("Hello there.", "".join(tap.text))

        await self.run_client(None, scenario)


class StockClientAgainstTheCascadeTests(unittest.IsolatedAsyncioTestCase):
    """The stock client, with a tool, against the real cascade adapter and the real core (scripted models, fake speech).

    The talker, the writer and the core's backend LLM share one scripted model client, as one model server would.
    """

    async def asyncSetUp(self):
        self.stt, self.tts = FakeSTT(), FakeTTS()

        async def factory(config, transport, connection):
            return cascade_parts(
                config, transport, connection, chat=self.chat, stt=self.stt, tts=self.tts, instructions=INSTRUCTIONS
            )

        async def core():  # the core is built once the test has its script, as the root builds it on first use
            built = chat_core(self.chat)
            self.addAsyncCleanup(built.aclose)
            return await core_hook(built)()

        app = FastAPI()
        mount_live(app, factory=factory, core=core)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self.serving = asyncio.create_task(self.server.serve())
        async with asyncio.timeout(10):
            while not self.server.started:
                await asyncio.sleep(0.02)
        self.port = self.server.servers[0].sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        self.server.should_exit = True
        async with asyncio.timeout(10):
            await self.serving

    async def test_a_delegated_request_runs_the_clients_tool_and_is_answered_aloud(self):
        self.chat = FakeChat(
            _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
            tool_call_round(),
            [chunk("We have Earl Grey and Matcha.", finish="stop")],
            _decision_chunks('{"action":"speak","speech":"We have Earl Grey and Matcha."}'),
        )
        calls: list[dict] = []

        async def get_menu(params):
            calls.append(params.arguments)
            await params.result_callback({"teas": ["Earl Grey", "Matcha"]})

        tools = ToolsSchema(
            [
                FunctionSchema(
                    "get_menu",
                    "Look up the menu.",
                    {"category": {"type": "string"}},
                    [],
                    handler=get_menu,
                )
            ]
        )
        service = OpenAILiveLLMService(
            api_key="test-token",
            base_url=f"ws://127.0.0.1:{self.port}/v1/live/sessions",
            settings=OpenAILiveLLMService.Settings(model="live-1", system_instruction=INSTRUCTIONS),
            delegation=OpenAILiveLLMService.ResponsesDelegation(
                settings=OpenAIResponsesLLMService.Settings(model="org/org/model")
            ),
        )
        tap = Tap()
        started = asyncio.Event()
        service.add_event_handler("on_session_started", lambda *_: started.set())
        worker = PipelineWorker(Pipeline([service, tap]), enable_rtvi=False, idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            context = LLMContext([{"role": "system", "content": INSTRUCTIONS}], tools=tools)
            await worker.queue_frame(LLMContextFrame(context))
            async with asyncio.timeout(10):
                await started.wait()
            await self.stt.say("What teas do you have?")
            async with asyncio.timeout(20):
                while "Matcha." not in "".join(tap.text):
                    await asyncio.sleep(0.05)
            self.assertEqual(calls, [{"category": "tea"}])
            spoken = "".join(tap.text)
            self.assertIn("Let me check.", spoken)
            self.assertIn("We have Earl Grey and Matcha.", spoken)
            await worker.queue_frame(EndFrame())
            async with asyncio.timeout(15):
                await running
        finally:
            if not running.done():
                await worker.cancel()
                await asyncio.gather(running, return_exceptions=True)

    async def test_a_stock_client_with_no_backend_declines_each_client_delegation_aloud(self):
        # Pipecat's default, delegation=None, asks for client delegation and declines each one with commentary that
        # names it (Pipecat's services/openai/live/llm.py, _handle_client_delegation); the cascade says it.
        self.chat = FakeChat(
            _decision_chunks('{"action":"delegate","speech":"Let me check."}'),
            _decision_chunks('{"action":"speak","speech":"Sorry, I cannot look that up right now."}'),
        )
        created: list = []
        service = OpenAILiveLLMService(
            api_key="test-token",
            base_url=f"ws://127.0.0.1:{self.port}/v1/live/sessions",
            settings=OpenAILiveLLMService.Settings(model="live-1", system_instruction=INSTRUCTIONS),
            delegation=None,
        )
        tap = Tap()
        started = asyncio.Event()
        service.add_event_handler("on_session_started", lambda *_: started.set())
        service.add_event_handler("on_delegation_created", lambda _service, delegation: created.append(delegation))
        worker = PipelineWorker(Pipeline([service, tap]), enable_rtvi=False, idle_timeout_secs=None)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        running = asyncio.create_task(runner.run())
        try:
            await worker.queue_frame(LLMContextFrame(LLMContext([{"role": "system", "content": INSTRUCTIONS}])))
            async with asyncio.timeout(10):
                await started.wait()
            await self.stt.say("Is my order ready?")
            async with asyncio.timeout(20):
                while "right now." not in "".join(tap.text):
                    await asyncio.sleep(0.05)
            self.assertEqual([(d.target, d.response_id) for d in created], [("client", None)])
            update = self.chat.requests[1]["messages"][-1]["content"]
            self.assertIn("No backend is available to handle delegated work in this session.", update)
            self.assertIn('you already told the caller: "Let me check."', update)
            self.assertEqual(len(self.chat.requests), 2)  # the talker and the writer: the server runs no backend
            await worker.queue_frame(EndFrame())
            async with asyncio.timeout(15):
                await running
        finally:
            if not running.done():
                await worker.cancel()
                await asyncio.gather(running, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
