# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that only audible output from an observed tool-triggered run is linked."""

# ruff: noqa: D101, D102

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService

from nemotron_voice_agent.pipecat.shared.perf_diagnostics import MAX_ERROR_MESSAGE_CHARS, PerfDiagnosticsObserver


class ToolAttributionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = LLMContext()
        self.llm = LLMService()
        self.assistant = LLMAssistantAggregator(self.context)
        self.output = SimpleNamespace(name="output")
        self.tts = SimpleNamespace(name="tts")
        self.user = SimpleNamespace(name="user")
        self.observer = PerfDiagnosticsObserver(self.output, AsyncMock())

    async def push(self, frame, source, destination, *, upstream=False, first=True):
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=destination,
                frame=frame,
                direction=FrameDirection.UPSTREAM if upstream else FrameDirection.DOWNSTREAM,
                timestamp=1,
                first_push=first,
            )
        )

    async def tool_start(self, call="tool-1", *, group=None):
        await self.push(FunctionCallInProgressFrame("lookup", call, {}, group_id=group), self.llm, self.tts)

    async def tool_result(self, call="tool-1", *, result="found", **options):
        frame = FunctionCallResultFrame("lookup", call, {}, result, **options)
        await self.push(frame, self.llm, self.tts)
        await self.push(frame, self.output, self.assistant, first=False)

    async def run_context(self, *, tool_trigger=True, skip_tts=False):
        context = LLMContextFrame(self.context)
        if tool_trigger:
            await self.push(context, self.assistant, self.output, upstream=True)
            await self.push(context, self.tts, self.llm, upstream=True, first=False)
        else:
            await self.push(context, self.user, self.llm)
        start = LLMFullResponseStartFrame()
        start.skip_tts = skip_tts
        await self.push(start, self.llm, self.tts)
        await self.push(start, self.output, self.assistant, first=False)

    async def speak(self, epoch, *, nonempty=True):
        start = TTSStartedFrame(context_id=epoch)
        await self.push(start, self.tts, self.output)
        await self.push(start, self.output, self.assistant, first=False)
        if nonempty:
            audio = OutputAudioRawFrame(audio=b"\x00" * 800, sample_rate=1000, num_channels=1)
            await self.push(audio, self.output, self.assistant)
        await self.push(TTSStoppedFrame(context_id=epoch), self.output, self.assistant)

    def attributed_epochs(self):
        return {
            event["epoch"]: event["tool_call_ids"]
            for event in self.observer.pending
            if event["kind"] == "audio_epoch_start"
        }

    async def test_first_nonempty_epoch_consumes_link(self):
        await self.tool_start()
        await self.tool_result()
        await self.run_context()
        await self.speak("empty", nonempty=False)
        await self.speak("first-audio")
        await self.speak("second-audio")
        self.assertEqual(self.attributed_epochs(), {"first-audio": ["tool-1"], "second-audio": []})

    async def test_nontriggering_tool_results_do_not_tag_unrelated_context(self):
        cases = (
            {"run_llm": False},
            {"run_llm": True, "properties": FunctionCallResultProperties(run_llm=False)},
            {"properties": FunctionCallResultProperties(run_llm=True, is_final=False)},
            {"run_llm": True, "result": None},
        )
        for index, options in enumerate(cases):
            with self.subTest(options=options):
                call = f"tool-{index}"
                await self.tool_start(call)
                await self.tool_result(call, **options)
                await self.run_context()
                await self.speak(call)
                self.assertEqual(self.attributed_epochs()[call], [])

    async def test_default_group_waits_for_last_sibling(self):
        await self.tool_start("first", group="batch")
        await self.tool_start("second", group="batch")
        await self.tool_result("first")
        await self.run_context()
        await self.speak("independent")
        await self.tool_result("second")
        await self.run_context()
        await self.speak("tool-followup")
        self.assertEqual(self.attributed_epochs(), {"independent": [], "tool-followup": ["second"]})

    async def test_broadcast_records_tool_events_once(self):
        await self.tool_start()
        await self.push(FunctionCallInProgressFrame("lookup", "tool-1", {}), self.llm, self.user, upstream=True)
        await self.tool_result()
        await self.push(FunctionCallResultFrame("lookup", "tool-1", {}, "found"), self.llm, self.user, upstream=True)
        await self.run_context()
        await self.speak("tool-followup")
        kinds = [event["kind"] for event in self.observer.pending]
        self.assertEqual(kinds.count("tool_start"), 1)
        self.assertEqual(kinds.count("tool_end"), 1)
        self.assertEqual(self.attributed_epochs(), {"tool-followup": ["tool-1"]})

    async def test_user_context_does_not_consume_pending_tool_result(self):
        await self.tool_start()
        await self.tool_result()
        await self.run_context(tool_trigger=False)
        await self.speak("user-turn")
        await self.run_context()
        await self.speak("independent")
        self.assertEqual(self.attributed_epochs(), {"user-turn": [], "independent": []})

    async def test_turn_boundaries_and_cancel_discard_old_links(self):
        boundaries = (UserStartedSpeakingFrame(), InterruptionFrame(), FunctionCallCancelFrame("lookup", "tool-2"))
        for index, boundary in enumerate(boundaries):
            call = f"tool-{index}"
            await self.tool_start(call)
            await self.tool_result(call)
            await self.push(boundary, self.user, self.llm)
            await self.run_context()
            await self.speak(call)
        self.assertTrue(all(not calls for calls in self.attributed_epochs().values()))

    async def test_error_size_is_bounded_after_redaction(self):
        message = "Authorization: Bearer test-secret " + "x" * MAX_ERROR_MESSAGE_CHARS
        await self.push(ErrorFrame(message), self.llm, self.output)
        event = self.observer.pending[0]
        self.assertEqual(len(event["message"]), MAX_ERROR_MESSAGE_CHARS)
        self.assertNotIn("test-secret", event["message"])
