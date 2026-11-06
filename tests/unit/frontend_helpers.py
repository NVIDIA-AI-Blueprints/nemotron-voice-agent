# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Doubles of the frontend's tests: speech stand-ins, a scripted talker, a delegation part, and one whole call.

:class:`CallHarness` runs one call through the frontend's own build (``examples.frontend_backend_live.frontend``) on a
scripted model client, fake speech and an output that says when the bot speaks: step by step, or as a script of frames
that Pipecat's ``run_test`` plays (:meth:`CallHarness.play`). Its speech is :class:`SentenceTTS`, or :class:`ToneTTS`,
Pipecat's own text-to-speech service, where a test needs what that service does around the audio; its output is
:class:`TransportOutput`, Pipecat's own output transport, where a test needs how that output sends audio.
"""

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from openai.types.chat import ChatCompletion
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SystemFrame,
    TextType,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint, ReliabilityConfig
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.frontend import FrontendEdges, FrontendSpec, build_frontend
from examples.frontend_backend_live.media.turns import SmartTurn
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import LLM_SERVICES, RESPONSE_FORMATS
from tests.unit.cascade_helpers import FakeSTT, transcript_turns
from tests.unit.core_helpers import FakeChat, FakeStream, chunk, tool_fragment

PROMPTS = load_prompts()
ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
RELIABILITY = ReliabilityConfig(provider_retries=0, talker_attempt_timeout_seconds=2.0)
# The voice's settings in these tests: no progress line unless a test asks for one.
CONFIG = LiveConfig(
    slot="llm",
    reliability=RELIABILITY,
    delegation=LiveConfig(slot="s").delegation.model_copy(update={"progress_after_seconds": None}),
)
SEED = [{"role": "system", "content": "You are Remy."}]
SENTENCE = re.compile(r"\s*(.+?[.!?])\s+", re.S)


def decision(action: str, speech: str = "", *, size: int = 6, finish: str = "stop") -> list:
    """A streamed ``{action, speech}`` decision: the JSON in pieces of ``size`` characters, then its finish."""
    text = json.dumps({"action": action, "speech": speech})
    return [chunk(text[at : at + size]) for at in range(0, len(text), size)] + [chunk(finish=finish)]


def tool_call(name: str, arguments: str = "", *, call_id: str = "call_0", size: int = 8, finish="tool_calls") -> list:
    """A streamed reply that calls ``name``: its id and name, ``arguments`` in pieces of ``size``, then its finish."""
    pieces = [arguments[at : at + size] for at in range(0, len(arguments), size)]
    fragments = [tool_fragment(0, call_id, name, pieces[0] if pieces else "")]
    fragments += [tool_fragment(0, arguments=piece) for piece in pieces[1:]]
    return [chunk(tool_calls=[fragment]) for fragment in fragments] + [chunk(finish=finish)]


def word_turns(speech=CONFIG.speech) -> LLMUserAggregatorParams:
    """The user aggregator's settings with the caller's own turn start (``media/turns.py``).

    While the bot is silent, the caller's first sound starts the turn; while it speaks, their words do. A turn ends at
    the VAD's stop once its words are in, or with none at the stop timeout, as a turn end's silence ends it. Only the
    turn's start is the caller's turns' own: the caller speaks with :meth:`CallHarness.speak`.
    """
    start = SmartTurn(speech).user_params(welcome_enabled=False).user_turn_strategies.start
    return LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(start=start, stop=[ExternalUserTurnStopStrategy()]),
        user_turn_stop_timeout=0.3,
    )


def caller(text: str, seconds: float = 0.02) -> list:
    """A caller turn's frames: ``text`` said in ``seconds``, proposed as a speech service with turn detection does."""
    return [
        ProposedUserStartedSpeakingFrame(),
        SleepFrame(seconds),
        TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True),
        ProposedUserStoppedSpeakingFrame(),
    ]


@dataclass
class Act(SystemFrame):
    """A step of the test where it falls in a played script: the backend answers, the app appends, and so on."""

    action: Callable[[], object]


class Acts(FrameProcessor):
    """The call's input edge in these tests: it runs each :class:`Act` and passes every other frame."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, Act):
            frame.action()
        else:
            await self.push_frame(frame, direction)


class Interruptions(FrameProcessor):
    """After the output's reader: counts the interruptions that passed it, on their way to the assistant aggregator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passed = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame) and direction is FrameDirection.DOWNSTREAM:
            self.passed += 1
        await self.push_frame(frame, direction)


class SentenceTTS(FrameProcessor):
    """Text-to-speech stand-in that speaks per sentence: each sentence's audio (100 ms a word), then its text.

    A sentence in ``silent`` gets no audio. ``spoken`` lists every sentence it was given.
    """

    def __init__(self, *, silent=(), **kwargs):
        super().__init__(**kwargs)
        self.spoken: list[str] = []
        self.silent = set(silent)
        self._buffer = ""
        self._started = False

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
    """Pipecat's text-to-speech service, speaking per sentence as ``NvidiaTTSService`` does: ``word_ms`` of tone a word.

    Each response's sentences share one audio context, which a sentence's audio joins as it is synthesized; a sentence
    in ``silent`` gets none. Pipecat's base class does the rest as for the NVIDIA service: it announces each sentence
    ahead of its audio, queues its text behind the audio, ends a context that has had nothing for its stop timeout, and
    writes itself off after three contexts in a row with no audio, as the example's own text-to-speech does.
    ``spoken`` lists each sentence it was given, with the loop time.
    """

    def __init__(self, *, silent=(), word_ms: int = 100, **kwargs):
        super().__init__(
            push_stop_frames=True,
            sample_rate=24000,
            settings=TTSSettings(model=None, voice="tone", language=None),
            **kwargs,
        )
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
    """Output stand-in: plays each line at once and says when the bot starts and stops speaking.

    While ``playing`` is clear, a line's end waits, as a line still playing does, and so does any frame ``pause_at``
    picks (the audio of a sentence not played yet, say).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speaking = False
        self.playing = asyncio.Event()
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
    """Pipecat's own output transport, writing reply audio in real time in chunks of ``chunk_ms``.

    As every Pipecat output does, it sends audio in whole chunks and passes the frames between them in their place, so
    a sentence whose audio is shorter than a chunk has its text go out before the audio the next sentence fills up.
    """

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
    """A scripted model client that also answers a one-shot (not streamed) request, as ``run_inference`` reads one.

    A one-shot reply is a text step, or a streamed step's chunk texts joined, so one scripted wording serves a relay
    worded in the pipeline or early, out of band. An ``asyncio.Event`` in the script holds the next call until it is
    set; one among a step's chunks holds the rest of that stream.
    """

    async def create(self, **kwargs):
        if self.script and isinstance(self.script[0], asyncio.Event):
            await self.script.pop(0).wait()
        if kwargs.get("stream"):
            self.requests.append(kwargs)
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return HeldStream(step)
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, list):
            for item in step:
                if isinstance(item, asyncio.Event):
                    await item.wait()
            step = "".join(item.choices[0].delta.content or "" for item in step if hasattr(item, "choices"))
        message = {"role": "assistant", "content": step}
        return ChatCompletion.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            }
        )


class Port:
    """A delegation part: notes each delegation's input, accepts it as ``del_<n>``, or refuses it.

    With ``answer`` an ``asyncio.Event``, each delegation waits for it before the part answers.
    """

    target = "responses"

    def __init__(
        self,
        *,
        refuse: bool = False,
        reports_events: bool = True,
        acknowledgement_delay_ms: int = 0,
        answer: asyncio.Event | None = None,
    ):
        self.refuse, self.reports_events, self.acknowledgement_delay_ms = (
            refuse,
            reports_events,
            acknowledgement_delay_ms,
        )
        self.answer_event = answer
        self.asked = 0  # the delegations the part was asked for, answered or not
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

    def answer(self, delegation_id: str, *texts: str, item: str = "msg") -> None:
        """The core's events for a delegation answered with ``texts``, one ``final_answer`` each."""
        self.started(delegation_id)
        for index, text in enumerate(texts):
            self.result(delegation_id, text, f"{item}_{delegation_id}_{index}")
        completed = {"id": f"resp_{delegation_id}", "status": "completed"}
        self.listener.on_delegation_event(delegation_id, {"type": "response.completed", "response": completed})

    def result(self, delegation_id: str, text: str, item_id: str) -> None:
        """One ``final_answer`` of a delegation whose work runs."""
        done = {
            "type": "message",
            "id": item_id,
            "status": "completed",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": text}],
        }
        self.listener.on_delegation_event(delegation_id, {"type": "response.output_item.done", "item": done})

    def started(self, delegation_id: str) -> None:
        """The core's events for a delegation whose work runs."""
        response = {"id": f"resp_{delegation_id}", "status": "in_progress"}
        self.listener.on_delegation_event(delegation_id, {"type": "response.created", "response": response})
        self.listener.on_delegation_event(delegation_id, {"type": "response.in_progress", "response": response})

    def waits_for_app(self, delegation_id: str) -> None:
        """The core's events for a delegation whose response ends with a call the application runs."""
        self.started(delegation_id)
        call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_menu", "arguments": "{}"}
        self.listener.on_delegation_event(delegation_id, {"type": "response.output_item.done", "item": call})
        completed = {"id": f"resp_{delegation_id}", "status": "completed"}
        self.listener.on_delegation_event(delegation_id, {"type": "response.completed", "response": completed})

    def fail(self, delegation_id: str) -> None:
        """The core's events for a delegation that failed before any tool ran."""
        response = {"id": f"resp_{delegation_id}", "status": "in_progress"}
        self.listener.on_delegation_event(delegation_id, {"type": "response.created", "response": response})
        self.listener.on_delegation_event(delegation_id, {"type": "response.in_progress", "response": response})
        failed = {**response, "status": "failed", "error": {"code": "server_error", "message": "boom"}}
        self.listener.on_delegation_event(delegation_id, {"type": "response.failed", "response": failed})

    def text(self, index: int = -1) -> str:
        """The transcript a delegation's input carried."""
        return self.inputs[index][0]["content"][0]["text"]


class CallHarness:
    """One call through the frontend's own build, on a scripted model client and fake speech.

    A test drives it step by step inside ``async with``, or plays a script of frames with :meth:`play`. The talker is
    the service that ``style`` names, as the hosts pick the style the cascade's ``talker_style`` names for the model.
    """

    def __init__(
        self, *script, config=CONFIG, port=None, seed=SEED, tts=None, user_params=None, output=None, style="json"
    ):
        self.chat = ScriptedChat(*script)
        self.config, self.port = config, port or Port()
        level = pick(RESPONSE_FORMATS, ENDPOINT.response_format, "response_format")
        service = pick(LLM_SERVICES, style, "talker_style")
        self.talker = service.from_config(ENDPOINT, config.reliability, level, client=self.chat)
        self.stt, self.tts, self.output = FakeSTT(), tts or SentenceTTS(), output or SpeakingOutput()
        self.interruptions = Interruptions()
        # A turn with no words ends soon after its speech, as a turn end's silence ends it.
        params = user_params or replace(transcript_turns(), user_turn_stop_timeout=0.3)
        spec = FrontendSpec(
            config=config,
            talker=self.talker,
            prompts=PROMPTS.cascade,
            lines=PROMPTS.lines,
            seed=list(seed),
            stt=self.stt,
            tts=self.tts,
            user_params=params,
            delegation=self.port,
            item_text=message_text,
        )
        edges = FrontendEdges(
            inputs=[Acts()], outputs=[self.output], after=[self.interruptions], worker={"idle_timeout_secs": None}
        )
        built = build_frontend(spec, edges)
        self.worker, self.voice, self.context = built.worker, built.voice, built.context
        self.processors, self.assistant = built.processors, built.assistant_aggregator
        self.errors: list = []

        @self.worker.event_handler("on_pipeline_error")
        async def on_error(worker, frame):
            self.errors.append(frame)

    async def __aenter__(self):
        """Run the call's pipeline and start the voice."""
        started = asyncio.Event()

        @self.worker.event_handler("on_pipeline_started")
        async def on_started(worker, frame):
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self.run = asyncio.create_task(runner.run())
        async with asyncio.timeout(5):
            await started.wait()
        self.voice.start()
        return self

    async def __aexit__(self, *exc):
        """Close the voice and stop the pipeline."""
        await self.voice.close()
        await self.worker.cancel()
        await asyncio.gather(self.run, return_exceptions=True)

    async def play(self, *frames) -> None:
        """Play the call with Pipecat's ``run_test``: ``frames`` enter the pipeline in order, then its end.

        A ``SleepFrame`` pauses the script, and an :class:`Act` runs a step of the test where it falls. The voice starts
        before the first frame and closes once the pipeline has ended, so the checks read the call as it ended.
        """
        self.voice.start()
        try:
            await run_test(Pipeline(self.processors), frames_to_send=frames)
        finally:
            await self.voice.close()

    async def say(self, text: str) -> None:
        """The caller says ``text``: a turn with its words, proposed as a speech service with turn detection does.

        The words and the turn's end come once the cut its start made has reached the output's reader, as in a call,
        where a turn lasts far longer than the cut takes to get there.
        """
        await self._start_turn()
        words = TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
        await self.stt.push_frame(words)
        await self.stt.push_frame(ProposedUserStoppedSpeakingFrame())

    async def noise(self) -> None:
        """The caller makes a sound with no words: a turn that ends empty once its cut has reached the output."""
        await self._start_turn()
        await self.stt.push_frame(ProposedUserStoppedSpeakingFrame())

    async def speak(self, *heard: str) -> None:
        """The caller speaks as the speech-to-text hears them: the VAD's start, ``heard`` in turn, then the VAD's stop.

        Each of ``heard`` but the last is an interim transcript and the last is the final one; with none, the caller
        makes a sound with no words, such as a cough. Only a harness on :func:`word_turns` starts its turns from these.
        """
        await self.stt.push_frame(VADUserStartedSpeakingFrame(start_secs=0.12))
        for index, text in enumerate(heard):
            await asyncio.sleep(0.05)  # each transcript in its turn, after the speech it comes from
            if index < len(heard) - 1:
                await self.stt.push_frame(InterimTranscriptionFrame(text=text, user_id="caller", timestamp="0"))
            else:
                final = TranscriptionFrame(text=text, user_id="caller", timestamp=str(time.time()), finalized=True)
                await self.stt.push_frame(final)
        await asyncio.sleep(0.05)
        await self.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))

    async def _start_turn(self) -> None:
        """The caller starts speaking; return once the interruption it makes has passed the output's reader."""
        passed = self.interruptions.passed
        await self.stt.push_frame(ProposedUserStartedSpeakingFrame())
        await self.until(lambda: self.interruptions.passed > passed, "the caller's interruption past the output")

    async def until(self, predicate, what: str, timeout: float = 4.0) -> None:
        """Wait until ``predicate()`` holds."""
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    await asyncio.sleep(0.01)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}") from None

    async def quiet(self, settle: float = 0.15) -> None:
        """Wait until nothing is due, no call runs and no line is in flight, then a moment more."""
        await self.until(lambda: self.voice.call is None and not self.voice._due(), "the voice to be idle")
        await asyncio.sleep(settle)

    def said(self) -> list[str]:
        """The assistant's lines in the talker's context, in order: its routing tools' calls are not lines."""
        messages = self.context.get_messages()
        return [m["content"] for m in messages if m.get("role") == "assistant" and not m.get("tool_calls")]

    def request(self, index: int = -1) -> dict:
        return self.chat.requests[index]
