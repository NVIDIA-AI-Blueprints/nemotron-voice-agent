# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Shared doubles for the live session protocol tests."""

import asyncio

from live.protocol import ProtocolError, SessionConfig


class FakeEngine:
    """A LiveEngine that records what the protocol layer asks of it."""

    def __init__(self, fail_start=False):
        self.fail_start = fail_start
        self.session = None
        self.started = self.closed = False
        self.muted = False
        self.audio: list[bytes] = []
        self.appended: list[tuple] = []
        self.items: list[dict] = []
        self.creates = 0
        self.updates: list[SessionConfig] = []
        self.delegations = {"item_known"}
        self.models_rejected = {"bad-model"}
        self.item_error: ProtocolError | None = None

    async def start(self, session):
        if self.fail_start:
            raise RuntimeError("pipeline failed")
        self.session, self.started = session, True

    async def close(self):
        self.closed = True

    def validate_backend_model(self, model):
        if model in self.models_rejected:
            raise ValueError("model not served")

    def update_backend(self, config):
        self.updates.append(config)

    def knows_delegation(self, delegation_id):
        return delegation_id in self.delegations

    async def set_muted(self, muted):
        self.muted = muted

    async def feed_audio(self, pcm24k):
        self.audio.append(pcm24k)

    async def append_instructions(self, text):
        self.appended.append(("instructions", text))

    async def append_thinking(self, text):
        self.appended.append(("thinking", text))

    async def append_commentary(self, text, delegation_id):
        self.appended.append(("commentary", text, delegation_id))

    def item_create(self, item):
        if self.item_error:
            raise self.item_error
        self.items.append(item)

    async def response_create(self):
        self.creates += 1


class Collector:
    """A subscriber that keeps every event it receives."""

    def __init__(self):
        self.events: list[dict] = []
        self._arrived = asyncio.Event()

    async def send(self, message):
        self.events.append(message)
        self._arrived.set()

    async def wait_for(self, kind, timeout=2.0):
        async with asyncio.timeout(timeout):
            while True:
                for event in self.events:
                    if event["type"] == kind:
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def kinds(self):
        return [event["type"] for event in self.events]


def responses_config(**kwargs):
    """A responses-mode session configuration."""
    return SessionConfig(
        model="live-1",
        delegation={
            "type": "responses",
            "responses": {"model": "org/org/model", "reasoning": {"effort": "low", "summary": "auto"}},
        },
        **kwargs,
    )


# ------------------------------------------------------------- Pipecat fakes for engine tests
import json  # noqa: E402
import time  # noqa: E402

from pipecat.frames.frames import (  # noqa: E402
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor  # noqa: E402


class FakeSTT(FrameProcessor):
    """Speech-to-text stand-in: the test says a sentence instead of sending speech audio."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def say(self, text):
        await self.push_frame(UserStartedSpeakingFrame())
        await self.push_frame(
            TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        )
        await self.push_frame(UserStoppedSpeakingFrame())


class FakeTTS(FrameProcessor):
    """Text-to-speech stand-in: 100 ms of tone per word, with the word frames a real TTS emits."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spoken: list[str] = []
        self._started = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            if not self._started:
                self._started = True
                await self.push_frame(TTSStartedFrame())
            for word in frame.text.split():
                self.spoken.append(word)
                await self.push_frame(TTSAudioRawFrame(audio=b"\x10\x10" * 2400, sample_rate=24000, num_channels=1))
                await self.push_frame(TTSTextFrame(text=word, aggregated_by="word"))
            return
        if isinstance(frame, LLMFullResponseEndFrame) and self._started:
            self._started = False
            await self.push_frame(TTSStoppedFrame())
        await self.push_frame(frame, direction)


class FakeWebSocket:
    """The server side of a primary WebSocket, driven by the test."""

    def __init__(self):
        self.to_server: asyncio.Queue = asyncio.Queue()
        self.from_server: list[dict] = []
        self._arrived = asyncio.Event()
        self.closed = False

    # What the gateway calls ---------------------------------------------------------------
    headers: dict = {}  # noqa: RUF012

    async def accept(self):
        return None

    async def receive_json(self):
        return json.loads(await self.to_server.get())

    async def receive_text(self):
        return await self.to_server.get()

    async def send_json(self, message):
        self.from_server.append(message)
        self._arrived.set()

    async def close(self, code=1000):
        self.closed = True

    # What the test calls --------------------------------------------------------------------
    def send(self, message):
        self.to_server.put_nowait(json.dumps(message))

    async def wait_for(self, kind, *, after=0, timeout=10.0, where=None):
        async with asyncio.timeout(timeout):
            while True:
                for event in self.from_server[after:]:
                    if event["type"] == kind and (where is None or where(event)):
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def kinds(self, skip_audio=True):
        return [e["type"] for e in self.from_server if not (skip_audio and e["type"] == "session.output_audio.delta")]
