# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve benchmark raw-token provenance independently of played TTS text."""

import unittest
from types import SimpleNamespace

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMTextFrame, LLMThoughtTextFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection

from examples.generic.offline.incremental import _Scheduled, _Session
from examples.generic.offline.native_response import NativeProviderFrame


class RawLLMEventTests(unittest.IsolatedAsyncioTestCase):
    """The public evidence distinguishes provider tokens from other frame text."""

    async def test_text_is_verbatim_and_only_from_the_downstream_llm(self):
        """Do not replace raw tokens with normalized TTS, thought, or upstream text."""
        events = []
        session = SimpleNamespace(
            active={"id": "response-1"},
            emit=lambda kind, **details: events.append({"type": kind, **details}),
        )
        observations = [
            NativeProviderFrame(LLMTextFrame(text="\n A  \u03b2 "), FrameDirection.DOWNSTREAM, "llm"),
            NativeProviderFrame(LLMTextFrame(text="upstream"), FrameDirection.UPSTREAM, "llm"),
            NativeProviderFrame(
                TTSTextFrame(text="normalized", aggregated_by="sentence"), FrameDirection.DOWNSTREAM, "tts"
            ),
            NativeProviderFrame(LLMThoughtTextFrame(text="private reasoning"), FrameDirection.DOWNSTREAM, "llm"),
            NativeProviderFrame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM, "llm"),
        ]
        for index, observation in enumerate(observations):
            await _Session.deliver(session, _Scheduled(1234, index, "provider_observation", observation, "response-1"))

        self.assertEqual(events[0]["text"], "\n A  \u03b2 ")
        self.assertEqual(events[0]["response_id"], "response-1")
        self.assertEqual(events[0]["source"], "llm")
        self.assertEqual(events[0]["direction"], "DOWNSTREAM")
        self.assertTrue(all("text" not in event for event in events[1:]))
        self.assertEqual(events[1]["direction"], "UPSTREAM")
        self.assertEqual(events[2]["source"], "tts")
        self.assertEqual(events[-1]["frame"], "LLMFullResponseEndFrame")

    async def test_stale_generation_never_exposes_candidate_tokens(self):
        """A cancelled generation cannot become a later candidate through evidence."""
        events = []
        session = SimpleNamespace(
            active={"id": "response-2"},
            emit=lambda kind, **details: events.append({"type": kind, **details}),
        )
        frame = NativeProviderFrame(LLMTextFrame(text="old answer"), FrameDirection.DOWNSTREAM, "llm")
        await _Session.deliver(session, _Scheduled(1234, 0, "provider_observation", frame, "response-1"))
        self.assertEqual(events, [{"type": "stale_provider_frame_discarded", "response_id": "response-1"}])
