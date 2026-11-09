# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Doubles of the frontend's tests: one call through the frontend's own build, the adapter's edges, and the roots."""

from __future__ import annotations

import asyncio
import json
import re
import time
import unittest
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, ClassVar
from unittest.mock import patch

from openai.types.chat import ChatCompletion
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner
from starlette.websockets import WebSocketState

import examples_registry
from examples.frontend_backend_live import factory
from examples.frontend_backend_live.adapter.contract import ServiceStarting
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint, ReliabilityConfig
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.frontend import FrontendEdges, FrontendSpec, build_frontend
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import LLM_SERVICES, RESPONSE_FORMATS
from gateways.live.protocol import AudioFormat
from tests.unit.cascade_helpers import FakeSTT, scripted_talker, transcript_turns
from tests.unit.core_helpers import FakeChat, FakeStream, chunk, tool_fragment

PROMPTS = load_prompts()
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
RELIABILITY = ReliabilityConfig(provider_retries=0, talker_attempt_timeout_seconds=2.0)
CONFIG = LiveConfig(
    slot="llm",
    reliability=RELIABILITY,
    delegation=LiveConfig(slot="s").delegation.model_copy(update={"progress_after_seconds": None}),
)
SEED = [{"role": "system", "content": "You are Remy."}]
SENTENCE = re.compile(r"\s*(.+?[.!?])\s+", re.S)


def decision(action: str, speech: str = "", *, size: int = 6, finish: str = "stop") -> list:
    """Return a streamed ``{action, speech}`` decision: its JSON in pieces of ``size`` characters, then its finish."""
    text = json.dumps({"action": action, "speech": speech})
    return [chunk(text[at : at + size]) for at in range(0, len(text), size)] + [chunk(finish=finish)]


async def wait_until(predicate, what: str, timeout: float = 4.0) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)
    except TimeoutError:
        raise AssertionError(f"timed out waiting for {what}") from None


def tool_call(name: str, arguments: str = "", *, call_id: str = "call_0", size: int = 8, finish="tool_calls") -> list:
    """Return a streamed reply that calls ``name``, with ``arguments`` in pieces of ``size``, then its finish."""
    pieces = [arguments[at : at + size] for at in range(0, len(arguments), size)]
    fragments = [tool_fragment(0, call_id, name, pieces[0] if pieces else "")]
    fragments += [tool_fragment(0, arguments=piece) for piece in pieces[1:]]
    return [chunk(tool_calls=[fragment]) for fragment in fragments] + [chunk(finish=finish)]


class Interruptions(BaseObserver):
    """Observer counting the interruptions the assistant aggregator passed on, once it saved a cut line's words."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passed = 0

    async def on_push_frame(self, data: FramePushed):
        frame, source = data.frame, data.source
        if isinstance(frame, InterruptionFrame) and isinstance(source, LLMAssistantAggregator):
            self.passed += data.direction is FrameDirection.DOWNSTREAM


class SentenceTTS(FrameProcessor):
    """Per-sentence text-to-speech stand-in: audio (100 ms a word), then text; a ``silent`` sentence gets no audio."""

    def __init__(self, *, silent=(), **kwargs):
        super().__init__(**kwargs)
        self.spoken: list[str] = []
        self.silent = set(silent)
        self._buffer, self._started = "", False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self._buffer += frame.text
            while match := SENTENCE.match(self._buffer):
                self._buffer = self._buffer[match.end() :]
                await self._say(match.group(1))
            return
        if isinstance(frame, InterruptionFrame):
            self._buffer, self._started = "", False
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buffer.strip():
                await self._say(self._buffer.strip())
            self._buffer = ""
            if self._started:
                self._started = False
                await self.push_frame(TTSStoppedFrame())
        await self.push_frame(frame, direction)

    async def _say(self, sentence: str) -> None:
        if not self._started:
            self._started = True
            await self.push_frame(TTSStartedFrame())
        self.spoken.append(sentence)
        if sentence not in self.silent:
            audio = b"\x10\x10" * 2400 * len(sentence.split())
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1))
        await self.push_frame(TTSTextFrame(text=sentence, text_type=TextType.SENTENCE))


class ToneTTS(TTSService):
    """Pipecat's ``TTSService``, with its stop timeout and no-audio limit, speaking ``word_ms`` of tone a word."""

    def __init__(self, *, silent=(), word_ms: int = 100, **kwargs):
        settings = TTSSettings(model=None, voice="tone", language=None)
        super().__init__(push_stop_frames=True, sample_rate=24000, settings=settings, **kwargs)
        self.silent, self.word_ms = set(silent), word_ms
        self.spoken: list[tuple[float, str]] = []

    async def run_tts(self, text, context_id):
        sentence = text.strip()
        self.spoken.append((asyncio.get_running_loop().time(), sentence))
        if not self.audio_context_available(context_id):
            await self.create_audio_context(context_id)
            yield TTSStartedFrame(context_id=context_id)
        if sentence not in self.silent:
            samples = 24 * self.word_ms * len(sentence.split())
            yield TTSAudioRawFrame(b"\x10\x10" * samples, 24000, 1, context_id=context_id)


class SpeakingOutput(FrameProcessor):
    """Output stand-in that plays each line at once; a line's end waits there while ``playing`` is clear."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speaking, self.playing = False, asyncio.Event()
        self.playing.set()
        self.pause_at = lambda frame: False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSStartedFrame) and not self.speaking:
                self.speaking = True
                await self.broadcast_frame(BotStartedSpeakingFrame)
            elif isinstance(frame, TTSStoppedFrame) or self.pause_at(frame):
                await self.playing.wait()
            elif isinstance(frame, InterruptionFrame) and self.speaking:
                self.speaking = False
                await self.push_frame(frame, direction)
                await self.broadcast_frame(BotStoppedSpeakingFrame)
                return
        await self.push_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame) and self.speaking:
            self.speaking = False
            await self.broadcast_frame(BotStoppedSpeakingFrame)


class TransportOutput(BaseOutputTransport):
    """Pipecat's own output transport, writing reply audio in real time in chunks of ``chunk_ms``."""

    def __init__(self, *, chunk_ms: int = 200, **kwargs):
        chunks = chunk_ms // 10
        params = TransportParams(audio_out_enabled=True, audio_out_sample_rate=24000, audio_out_10ms_chunks=chunks)
        super().__init__(params, **kwargs)

    async def start(self, frame):
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame) -> bool:
        await asyncio.sleep(len(frame.audio) / (frame.sample_rate * 2 * frame.num_channels))
        return True


class HeldStream(FakeStream):
    """A scripted stream that waits at each ``asyncio.Event`` among its chunks until it is set."""

    async def _iterate(self):
        for item in self.chunks:
            if isinstance(item, asyncio.Event):
                await item.wait()
            else:
                yield item


class ScriptedChat(FakeChat):
    """Scripted model client: an ``asyncio.Event`` holds what follows it; a one-shot request gets the step's text."""

    async def create(self, **kwargs):
        if self.script and isinstance(self.script[0], asyncio.Event):
            await self.script.pop(0).wait()
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if kwargs.get("stream"):
            return HeldStream(step)
        if isinstance(step, list):
            for item in step:
                if isinstance(item, asyncio.Event):
                    await item.wait()
            step = "".join(item.choices[0].delta.content or "" for item in step if hasattr(item, "choices"))
        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": step}}
        return ChatCompletion.model_validate(
            {"id": "c", "object": "chat.completion", "created": 0, "model": "m", "choices": [choice]}
        )


class Port:
    """A delegation part: notes each delegation's input and accepts it as ``del_<n>``, after ``answer`` if set."""

    target = "responses"

    def __init__(self, *, refuse=False, reports_events=True, acknowledgement_delay_ms=0, answer=None):
        self.refuse, self.reports_events = refuse, reports_events
        self.acknowledgement_delay_ms, self.answer_event = acknowledgement_delay_ms, answer
        self.asked = 0
        self.inputs: list[list[dict]] = []
        self.listener = None

    def listen(self, listener):
        self.listener = listener

    async def delegate(self, input):
        self.asked += 1
        if self.answer_event is not None:
            await self.answer_event.wait()
        if self.refuse:
            raise RuntimeError("refused")
        self.inputs.append(input)
        return f"del_{len(self.inputs)}"

    def send(self, delegation_id: str, kind: str, **fields) -> None:
        self.listener.on_delegation_event(delegation_id, {"type": kind, **fields})

    def answer(self, delegation_id: str, *texts: str) -> None:
        self.started(delegation_id)
        for index, text in enumerate(texts):
            self.result(delegation_id, text, f"msg_{delegation_id}_{index}")
        self.send(delegation_id, "response.completed", response={"id": f"resp_{delegation_id}", "status": "completed"})

    def result(self, delegation_id: str, text: str, item_id: str) -> None:
        content = [{"type": "output_text", "text": text}]
        item = {"type": "message", "id": item_id, "status": "completed", "phase": "final_answer", "content": content}
        self.send(delegation_id, "response.output_item.done", item=item)

    def started(self, delegation_id: str) -> None:
        response = {"id": f"resp_{delegation_id}", "status": "in_progress"}
        self.send(delegation_id, "response.created", response=response)
        self.send(delegation_id, "response.in_progress", response=response)

    def fail(self, delegation_id: str) -> None:
        self.started(delegation_id)
        error = {"code": "server_error", "message": "boom"}
        failed = {"id": f"resp_{delegation_id}", "status": "failed", "error": error}
        self.send(delegation_id, "response.failed", response=failed)

    def text(self, index: int = -1) -> str:
        return self.inputs[index][0]["content"][0]["text"]


class CallHarness:
    """One call through the frontend's own build, step by step; ``style`` names the talker, on a scripted client."""

    def __init__(self, *script, config=CONFIG, port=None, seed=SEED, tts=None, output=None, style="json"):
        self.chat = ScriptedChat(*script)
        self.config, self.port = config, port or Port()
        level = pick(RESPONSE_FORMATS, ENDPOINT.response_format, "response_format")
        service = pick(LLM_SERVICES, style, "talker_style")
        self.talker = scripted_talker(service, self.chat).from_config(ENDPOINT, config.reliability, level)
        self.stt, self.tts, self.output = FakeSTT(), tts or SentenceTTS(), output or SpeakingOutput()
        self.interruptions = Interruptions()
        spec = FrontendSpec(
            config=config,
            talker=self.talker,
            prompts=PROMPTS.cascade,
            lines=PROMPTS.lines,
            seed=list(seed),
            stt=self.stt,
            tts=self.tts,
            user_params=replace(transcript_turns(), user_turn_stop_timeout=0.3),  # a turn with no words ends soon
            delegation=self.port,
            item_text=message_text,
        )
        worker = {"idle_timeout_secs": None, "observers": [self.interruptions]}
        built = build_frontend(spec, FrontendEdges(inputs=[], outputs=[self.output], worker=worker))
        self.worker, self.voice, self.context = built.worker, built.voice, built.context
        self.processors, self.assistant = built.processors, built.assistant_aggregator
        self.errors: list = []
        self.worker.add_event_handler("on_pipeline_error", lambda worker, frame: self.errors.append(frame))

    async def __aenter__(self):
        started = asyncio.Event()
        self.worker.add_event_handler("on_pipeline_started", lambda worker, frame: started.set())
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self.run = asyncio.create_task(runner.run())
        async with asyncio.timeout(5):
            await started.wait()
        self.voice.start()
        return self

    async def __aexit__(self, *exc):
        await self.voice.close()
        await self.worker.cancel()
        await asyncio.gather(self.run, return_exceptions=True)

    async def say(self, text: str) -> None:
        """Say ``text`` as the caller, in a turn whose words come once its cut has passed the assistant aggregator."""
        await self._start_turn()
        words = TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        await self.stt.push_frame(words)
        await self.stt.push_frame(ProposedUserStoppedSpeakingFrame())

    async def noise(self) -> None:
        """Make a sound with no words as the caller: a turn that ends empty once its cut has passed the aggregator."""
        await self._start_turn()
        await self.stt.push_frame(ProposedUserStoppedSpeakingFrame())

    async def _start_turn(self) -> None:
        passed = self.interruptions.passed
        await self.stt.push_frame(ProposedUserStartedSpeakingFrame())
        await self.until(lambda: self.interruptions.passed > passed, "the caller's cut past the assistant aggregator")

    until = staticmethod(wait_until)

    async def quiet(self, settle: float = 0.15) -> None:
        """Wait until nothing is due, no call runs and no line is in flight, then a moment more."""
        await self.until(lambda: self.voice.call is None and not self.voice._due(), "the voice to be idle")
        await asyncio.sleep(settle)

    def said(self) -> list[str]:
        """Return the assistant's lines in the talker's context, in order, without the routing tools' calls."""
        messages = self.context.get_messages()
        return [m["content"] for m in messages if m.get("role") == "assistant" and not m.get("tool_calls")]

    def request(self, index: int = -1) -> dict:
        return self.chat.requests[index]


WIRE_RATE = AudioFormat().rate
PACKET_BYTES = WIRE_RATE // 10 * 2
FLUSH = object()  # a run_frames step that waits for the queued frames, so a cut after it cannot overtake them


def tone(ms: int, rate: int = WIRE_RATE, level: int = 0x1010) -> bytes:
    """Return ``ms`` of a constant PCM16 level at ``rate``."""
    return level.to_bytes(2, "little", signed=True) * (rate * ms // 1000)


class FakeSocket:
    """The gateway's primary socket as the adapter's transport reads it: :meth:`send` is what the client sends."""

    client_state = WebSocketState.CONNECTED

    def __init__(self, audio_format: AudioFormat | None = None):
        self.format = audio_format or AudioFormat()
        self.handed: list = []
        self.audio: dict = {}
        self.incoming: asyncio.Queue = asyncio.Queue()

    def send(self, text: str, audio: bytes | None = None) -> None:
        self.audio[text] = audio
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    async def receive(self):
        return await self.incoming.get()

    async def close(self, code=1000, reason=None):
        pass

    async def handle(self, data):
        self.handed.append(data)
        return self.audio.get(data)


class RecordingPort:
    """A frontend adapter's view of its session that notes what the adapter sends, in order."""

    def __init__(self, active=True, session_id="live_test", socket=None):
        self.id, self.input_ms, self.active = session_id, 0, active
        self.socket = socket or FakeSocket()
        self.log: list[tuple] = []

    def is_active(self):
        return self.active

    async def emit(self, kind, **fields):
        self.log.append(("emit", kind, fields))

    async def advance_input_clock(self, ms):
        self.input_ms = max(self.input_ms, ms)

    async def send_output_audio(self, audio):
        self.log.append(("audio", audio))

    async def report_error(self, message, client_event_id=None):
        self.log.append(("error", message))

    def request_close(self, reason):
        self.log.append(("close", reason))

    def packets(self):
        return [entry[1] for entry in self.log if entry[0] == "audio"]

    def sounds(self):
        """Return whether each packet carries sound, in order."""
        return [bool(any(audio)) for audio in self.packets()]

    def transcript(self):
        return [entry[2] for entry in self.log if entry[:2] == ("emit", "session.output_transcript.delta")]


class Probe(FrameProcessor):
    """Processor that records every frame that reaches it, with its direction, and passes it on."""

    def __init__(self):
        super().__init__()
        self.frames, self.directions = [], []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        self.directions.append(direction)
        await self.push_frame(frame, direction)

    def of(self, *kinds, direction=FrameDirection.DOWNSTREAM):
        return [f for f, d in zip(self.frames, self.directions, strict=True) if isinstance(f, kinds) and d is direction]


async def run_frames(processors, frames, *, settle=0.15, gap=0.02):
    """Run ``processors`` in a pipeline, queue ``frames`` ``gap`` apart, wait ``settle``, then cancel it."""
    worker = PipelineWorker(Pipeline(processors), idle_timeout_secs=None)
    started = asyncio.Event()
    worker.add_event_handler("on_pipeline_started", lambda worker, frame: started.set())
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    await asyncio.wait_for(started.wait(), timeout=5)
    for frame in frames:
        if frame is FLUSH:
            await worker.flush_pipeline()
        else:
            await worker.queue_frame(frame)
        await asyncio.sleep(gap)
    await asyncio.sleep(settle)
    await worker.cancel()
    await asyncio.gather(running, return_exceptions=True)


FRONT = {"model_id": "org/front", "base_url": "http://front/v1"}
CATALOG = ExampleCatalog(
    {
        "llm": [FRONT],
        "thinker-llm": [
            {"model_id": "org/back", "base_url": "http://back/v1"},
            {"model_id": "org/other", "base_url": "http://o/v1"},
        ],
        "asr": [{"server": "asr:50052"}],
        "tts": [{"server": "tts:50051"}],
    }
)


@dataclass(frozen=True)
class StubSettings:
    block: Mapping[str, Any]
    parsed: ClassVar[list] = []

    @classmethod
    def parse(cls, block: Mapping[str, Any], **helpers) -> StubSettings:
        cls.parsed.append(dict(block))
        return cls(dict(block))


class StubReadiness:
    """Readiness stand-in: ``down`` names the services still starting."""

    def __init__(self, down: tuple[str, ...] = ()):
        self.down, self.checks = down, 0

    async def ensure_ready(self) -> None:
        self.checks += 1
        if self.down:
            raise ServiceStarting(self.down[0])


class StubAdapter:
    """A frontend adapter class by the class contract; its instances only record how they were built."""

    Settings: ClassVar[type] = StubSettings
    built: ClassVar[list] = []

    def __init__(self, *args):
        self.args = args

    @classmethod
    def from_config(cls, session, transport, connection, settings, catalog):
        cls.built.append(cls(session, transport, connection, settings, catalog))
        return cls.built[-1]

    @classmethod
    def describe(cls, settings, catalog) -> dict:
        endpoint = catalog.model("llm")
        return {"frontend": {"provider": "stub-provider", "slot": "llm", "model": endpoint.model, "prompt": "generic"}}

    @classmethod
    def readiness(cls, settings, catalog) -> StubReadiness:
        return StubReadiness()


def reset_stubs() -> None:
    StubSettings.parsed.clear()
    StubAdapter.built.clear()


def clear_root_caches(case: unittest.TestCase) -> None:
    """Make the root read and build again, now and after the test."""
    for cached in (factory.root, factory.delegation_core, factory.responses_client):
        cached.cache_clear()
        case.addCleanup(cached.cache_clear)


class RootCase(unittest.TestCase):
    """A test of the root: it reads and builds again, with the stub adapter class as ``cascade`` and :data:`CATALOG`."""

    def setUp(self):
        clear_root_caches(self)
        reset_stubs()
        self.enterContext(patch.object(factory.ExampleCatalog, "read", return_value=CATALOG))
        self.enterContext(patch.object(factory, "frontend_adapters", return_value={"cascade": StubAdapter}))

    def with_settings(self, settings: dict) -> None:
        """Give the example's registry entry ``settings`` for the rest of the test."""
        self.enterContext(patch.dict(examples_registry.EXAMPLES[factory.EXAMPLE_KEY], {"config": settings}))


def stub_root(**changes) -> factory.Root:
    """Return a root state with the stub adapter class, the shipped settings and :data:`CATALOG`."""
    config = factory.example_config()
    core = factory.core_settings(config.core)
    settings = StubSettings({"display_label": core.delegation.display_label})
    values = {"config": config, "core": core, "adapter": StubAdapter, "settings": settings, "catalog": CATALOG}
    return factory.Root(**{**values, "readiness": StubReadiness(), **changes})


def module(name: str, **attributes) -> dict[str, ModuleType]:
    """Return a stand-in module for ``patch.dict(sys.modules, ...)``."""
    stand_in = ModuleType(name)
    stand_in.__dict__.update(attributes)
    return {name: stand_in}
