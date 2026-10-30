# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Bot goldens: whole calls through the repository UI bot, ``pipeline.bot()``, pinned channel by channel.

The bot has no Live gateway, so these goldens record what its own transport and services see. ``bot()`` runs as it
does for a browser session; only its outside edges are replaced:

* the transport (``create_transport``): :class:`FakeTransport`, whose input carries the client's RTVI messages into
  the pipeline and whose output keeps every message the bot sends the client;
* the speech services (``build_stt``, ``build_tts``) and the turn detection that listens to caller audio (the turn
  ends, ``TURN_ENDS``: Silero VAD and the Smart Turn model). ``FakeSTT`` says whole caller turns, so the external turn
  strategies of the whole-call tests end each turn when it has been said;
* the model clients: ``build_agent`` gets the talker's and the writer's scripted client, and ``build_core`` builds the
  bot's own core from its profile with the backend LLM on the backend's scripted client (``chat_adapters``);
* the services catalog (``ExampleCatalog.read``): :data:`OFFERED` in place of ``services.yaml``, ``settings.yaml`` and
  their endpoint probes;
* the environment: no audio recorder, no tracing, the welcome switch set by the test, no prompt path override.

Everything else is the bot's own code: its root (:func:`golden_root` runs ``bot_root``'s own reading of
``config.yaml`` and ``core.ui-bot.yaml``, so the prompt set and the voice note are the files' own), the cascade
(talker, writer, work note, the greeting line, the task cards), its delegation core and the cafe tools, which its
backend runs itself as server tools.

Timers: :func:`golden_root` sets the quiet before a progress line (``progress_after_seconds``) to an hour, so no golden
gets a clock-driven line. The other timers keep the files' values, which these calls never reach (a held round is
released well within its limit).

Channels, each compared on its own:

* ``spoken``: each utterance the text-to-speech received, in the order it spoke them;
* ``frontend_requests`` and ``backend_requests``: the arguments each scripted model client received;
* ``client_bot``, ``client_user`` and ``client_server``: the RTVI messages the bot sent the client, split by what they
  report: the bot (``bot-*``), the caller (``user-*``), and the bot's own server messages with any other type.
  Pipecat's ``metrics`` messages are left out: their values are timings.

The client messages are split because they come from separate tasks: Pipecat's RTVI observer reports frames that
different processors push, and the bot queues its ``user-turn-finalized`` message from an event handler's task, so
their order across the three channels is not fixed. Lines are normalized as in the harness (``Normalizer``): ids
become first-seen numbers, clock values ``<clock>``, the work note's seconds ``<seconds>``.

Writing a bot golden: subclass :class:`BotGoldenCase`, ``serve`` the scripts, act as the client (``client_ready``,
``say``), wait for quiet before each action, then ``close`` and ``assert_golden``. A :class:`Held` step keeps a model
request open until the test releases it, as a model that is still working, so the call can do something while the
backend works; ``wait_quiet`` treats a held backend request as quiet. Recording works as in the harness:
``LIVE_GOLDEN_RECORD=1`` writes ``recordings/<name>.json``; a normal run only compares. A test
that changes something ``serve`` also patches (``CascadeSession``, ``LiveTalkerProcessor``, the ``pipeline`` names)
patches it before ``serve`` with ``start()`` and ``addCleanup(stop)``, never in a ``with`` block around the call:
``serve`` undoes its own patches at clean-up, after such a block has ended, and would put the test's version back.

Bot internals the harness touches (re-point these when a change moves them):

* :meth:`BotGoldenCase.instrument` wraps ``CascadeSession.on_turn``, ``LiveTalkerProcessor.speak``,
  ``LiveTalkerProcessor.attach`` and ``IMSession.start`` with wrappers that call the original unchanged; the last
  records the IM session that the bot's core starts for its conversation;
* :meth:`BotGoldenCase.bot_pending` reads that IM session's ``busy`` and ``CascadeSession.presenter.busy``: the
  results the core handed over that the cascade has not spoken;
* :meth:`BotGoldenCase.replace_edges` hands the scripted clients to ``build_agent`` and to the bot's ``build_core``.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

from pipecat.frames.frames import (
    InputTransportMessageFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from delegation_core.app import build_core
from delegation_core.backend.tools.cafe.schemas import CAFE_TOOLS
from delegation_core.backend.tools.cafe.tools import CafeTools
from delegation_core.im.session import IMSession
from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.media.playout import PlayoutTracker
from examples.frontend_backend_live.media.processor import LiveTalkerProcessor
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from tests.unit.cascade_helpers import FakeSTT, chat_adapters
from tests.unit.goldens.harness import (
    GOLDEN_DIR,
    QUIET_PROGRESS_SECONDS,
    RECORD_VARIABLE,
    WAIT_SECONDS,
    GoldenCase,
    Normalizer,
    RecordingTTS,
    ScriptedChat,
    answer_round,
    canonical,
    channel_diff,
    chunk,
    decision,
    tool_call_round,
    tool_fragment,
)

BOT_CHANNELS = (
    "spoken",
    "frontend_requests",
    "backend_requests",
    "client_bot",
    "client_user",
    "client_server",
)

# What the browser client sends for a session of this example, less the model fields the catalog supplies.
SESSION_BODY = {"pipeline_mode": "frontend-backend-live", "prompt_key": "talker", "session_id": "golden"}

# The catalog entries the bot reads (ExampleCatalog), in place of services.yaml and settings.yaml: the example's
# default model for both roles, with its registry settings applied (examples_registry.yaml): reasoning off for the
# talker, on for the backend. The base URL is never contacted; the model clients are scripted.
SERVICE_ENTRIES = {
    "llm": {
        "model_id": "nvidia/nemotron-3-super-120b-a12b",
        "base_url": "http://localhost:1/v1",
        "extra_params": {
            "extra_body": {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": False}},
            "temperature": 0.0,
            "max_tokens": 512,
        },
    },
    "thinker-llm": {
        "model_id": "nvidia/nemotron-3-super-120b-a12b",
        "base_url": "http://localhost:1/v1",
        "extra_params": {
            "extra_body": {"repetition_penalty": 1.05, "chat_template_kwargs": {"enable_thinking": True}},
            "temperature": 0.7,
            "max_tokens": 4096,
        },
    },
}

# The example's catalog, by services slot: the talker's slot (llm) and the backend's (thinker-llm, core.yaml's
# backend.slot). The speech slots list nothing: the speech services are fakes.
OFFERED = {slot: [entry] for slot, entry in SERVICE_ENTRIES.items()}

# The deployment's welcome switch (examples_registry.welcome_message_enabled); each golden sets it.
WELCOME_VARIABLE = "ENABLE_WELCOME_MESSAGE"
# Variables that would point the bot at another prompt catalog or prompt (utils.resolve_prompt); removed for the test.
PROMPT_VARIABLES = ("PROMPT_FILE_PATH", "PROMPT_SELECTOR")

# The message the browser client (@pipecat-ai/client-js 1.13.1, client/package.json) sends once it is connected.
CLIENT_READY = {
    "label": "rtvi-ai",
    "type": "client-ready",
    "id": "ready",
    "data": {"version": "2.1.0", "about": {"library": "@pipecat-ai/client-js", "library_version": "1.13.1"}},
}
# Pipecat's timing reports; no channel keeps them.
METRICS_MESSAGE = "metrics"


# --------------------------------------------------------------------------------------- the bot's edges
READ_BOT_ROOT = pipeline.bot_root.__wrapped__  # bot_root's own reading, without its cache


def offered_catalog():
    """Stands in for ``ExampleCatalog.read``: the catalog of :data:`OFFERED`."""
    return ExampleCatalog(json.loads(json.dumps(OFFERED)))


def golden_root():
    """The bot's root state, read by ``bot_root``'s own code (uncached), with progress lines an hour away.

    A progress line may come after 8 s of silent work (``progress_after_seconds``); a golden that holds the backend for
    as long as it needs must not get one.
    """
    state = READ_BOT_ROOT()
    delegation = replace(state.cascade.delegation, progress_after_seconds=QUIET_PROGRESS_SECONDS)
    return replace(state, cascade=replace(state.cascade, delegation=delegation))


def restore_environment(saved):
    """Put back the variables ``saved`` maps to their values (None: the variable was not set)."""
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class ExternalTurns:
    """Stands in for every turn end (``TURN_ENDS``): a turn starts and ends when the fake speech-to-text says so.

    The bot's own turn end detects turns from caller audio (Silero VAD, Smart Turn) and mutes the caller until the
    welcome has been spoken; the fake speech-to-text sends no audio.
    """

    def __init__(self, speech):
        self.speech = speech

    def user_params(self, welcome_enabled):
        return LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()], stop=[ExternalUserTurnStopStrategy()]
            )
        )


# ------------------------------------------------------------------------------------------------- doubles
class Held:
    """A script step its client keeps waiting, as a model that is still working, until the test releases it."""

    def __init__(self, step):
        self.step = step
        self.released = asyncio.Event()

    def release(self):
        self.released.set()


class BotChat(ScriptedChat):
    """``ScriptedChat`` that can hold a request on a :class:`Held` step, and counts how often it is closed."""

    def __init__(self, role, *script, on_request=None):
        super().__init__(role, *script, on_request=on_request)
        self.holding: Held | None = None
        self.closes = 0

    @property
    def parked(self):
        """Whether a request is waiting on a step the test has not released."""
        return self.holding is not None and not self.holding.released.is_set()

    async def create(self, **kwargs):
        if self.script and isinstance(self.script[0], Held):
            held = self.holding = self.script[0]
            self.on_request()
            await held.released.wait()
            self.script[0], self.holding = held.step, None
        return await super().create(**kwargs)

    async def close(self):
        self.closes += 1
        await super().close()


class TransportInput(FrameProcessor):
    """The fake transport's input. It passes every frame on; the client's messages enter the pipeline here."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def receive(self, message):
        # As a real transport does, in both directions: upstream it reaches the RTVI processor at the pipeline's head.
        await self.broadcast_frame(InputTransportMessageFrame, message=message)


class TransportOutput(FrameProcessor):
    """The fake transport's output. It keeps every message the bot sends the client and passes the rest on."""

    def __init__(self, on_message, **kwargs):
        super().__init__(**kwargs)
        self.on_message = on_message
        self.messages: list[dict] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        is_message = isinstance(frame, OutputTransportMessageFrame | OutputTransportMessageUrgentFrame)
        if is_message and direction == FrameDirection.DOWNSTREAM:
            self.messages.append(json.loads(json.dumps(frame.message, default=repr)))
            self.on_message()
            return
        await self.push_frame(frame, direction)


class FakeTransport(BaseTransport):
    """Stands in for the browser transport ``create_transport`` builds (WebRTC or WebSocket)."""

    def __init__(self, on_message):
        super().__init__(name="FakeTransport")
        self._input = TransportInput(name="FakeTransport#input")
        self._output = TransportOutput(on_message, name="FakeTransport#output")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")

    def input(self):
        return self._input

    def output(self):
        return self._output

    @property
    def messages(self):
        return self._output.messages

    async def receive(self, message):
        """A message from the client."""
        await self._input.receive(message)

    async def disconnect(self):
        """The client leaves, as when the browser tab closes; the bot's handler cancels its pipeline."""
        await self._call_event_handler("on_client_disconnected", "client")


# --------------------------------------------------------------------------------------------- the golden
def normalize_bot_channels(raw):
    """Normalize every channel on its own, with the harness's normalizer."""
    return {channel: Normalizer()(raw[channel]) for channel in BOT_CHANNELS}


def changed_bot_channels(name, expected, recorded):
    """A unified diff for each channel whose JSON differs from ``expected`` (a golden file's channels)."""
    return [
        channel_diff(name, channel, expected.get(channel), recorded[channel])
        for channel in BOT_CHANNELS
        if canonical(expected.get(channel)) != canonical(recorded[channel])
    ]


def client_channel(message):
    """The channel a client message belongs to, or None for Pipecat's timing reports."""
    kind = str(message.get("type", ""))
    if kind == METRICS_MESSAGE:
        return None
    if kind.startswith("bot-"):
        return "client_bot"
    if kind.startswith("user-"):
        return "client_user"
    return "client_server"


class BotGoldenCase(unittest.IsolatedAsyncioTestCase):
    """Serves one scripted call per test through ``pipeline.bot()``; pins it to ``recordings/<name>.json``."""

    # The harness's generic parts, used unchanged: the bounded waits (``wait_quiet`` reads this class's ``pending``),
    # the script checks and the scenario text.
    wait_until = GoldenCase.wait_until
    wait_quiet = GoldenCase.wait_quiet
    check_scripts_not_overrun = GoldenCase.check_scripts_not_overrun
    check_scripts_used = GoldenCase.check_scripts_used
    scenario = GoldenCase.scenario

    # ---------------------------------------------------------------------------------------- starting
    async def serve(self, *, frontend=(), backend=(), welcome=True):
        """Run ``pipeline.bot()`` for one browser session; return once its pipeline has started.

        ``frontend`` lists the talker's and the writer's replies in the order they are asked for; ``backend`` lists
        the backend rounds in order. A step is what the harness's scripts take, or a :class:`Held` around one.
        ``welcome`` is the deployment's welcome switch (``ENABLE_WELCOME_MESSAGE``).
        """
        self.frontend = BotChat("frontend", *frontend, on_request=self._changed)
        self.backend = BotChat("backend", *backend, on_request=self._changed)
        self.stt, self.tts = FakeSTT(), RecordingTTS(on_change=self._changed)
        self.transport = FakeTransport(self._changed)
        self.welcome = welcome
        self.sessions: list[CascadeSession] = []  # the cascade the bot attaches to its processor
        self.ims: list[IMSession] = []  # the IM session the bot's core starts for its conversation
        self.workers: list[PipelineWorker] = []  # the pipeline worker the bot builds
        self.turns_expected = self.turns_started = self.turns_open = self.speak_calls = self.lines_said = 0
        self.started = False
        self._activity = asyncio.Event()
        patches = ExitStack()
        self.addCleanup(patches.close)  # cleanups run last-in first-out: the bot stops before the patches go
        self.instrument(patches)
        self.replace_edges(patches)
        self.tasks_before = asyncio.all_tasks()  # so close() can tell the tasks the bot started
        self.bot = asyncio.create_task(pipeline.bot(RunnerArguments(body=dict(SESSION_BODY))), name="bot")
        self.addAsyncCleanup(self.stop)
        await self.wait_until(lambda: self.started or self.bot.done(), "the bot's pipeline to start")
        if not self.started:
            self.fail(f"bot() ended before its pipeline started: {self.bot.exception()!r}")

    def replace_edges(self, patches: ExitStack):
        """Replace the bot's outside edges (see the module note); everything else runs as it does for a browser."""
        case = self

        def watched_worker(*args, **kwargs):
            # The bot's own worker, watched: the pipeline's start.
            worker = PipelineWorker(*args, **kwargs)

            @worker.event_handler("on_pipeline_started")
            async def on_started(worker, frame):
                case.started = True
                case._changed()

            case.workers.append(worker)
            return worker

        def scripted_agent(config, frontend_endpoint):
            return build_agent(config, frontend_endpoint, frontend_client=case.frontend)

        def scripted_core(settings, *, endpoints, defaults):
            # The bot's own core, from its profile, with its backend LLM on the backend's script.
            return build_core(settings, endpoints=endpoints, defaults=defaults, adapters=chat_adapters(case.backend))

        edges = {
            "bot_root": golden_root,
            "create_transport": lambda runner_args: case.transport,
            "build_stt": lambda body, entry, turn_end: case.stt,
            "build_tts": lambda body, entry: case.tts,
            "TURN_ENDS": dict.fromkeys(pipeline.TURN_ENDS, ExternalTurns),
            "build_agent": scripted_agent,
            "build_core": scripted_core,
            "create_audio_recorder": lambda: None,
            "IS_TRACING_ENABLED": False,
            "PipelineWorker": watched_worker,
        }
        for name, value in edges.items():
            patches.enter_context(patch.object(pipeline, name, value))
        patches.enter_context(patch.object(ExampleCatalog, "read", offered_catalog))
        # Only the files the bot names: no override of the example's settings file.
        patches.enter_context(patch.dict(os.environ))
        os.environ.pop("LIVE_CONFIG_PATH", None)
        # Only these variables are set and later restored. Restoring a snapshot of the whole environment at clean-up
        # would also bring back whatever the test itself had set at serve() time, after the test put it back.
        names = (*PROMPT_VARIABLES, WELCOME_VARIABLE)
        patches.callback(restore_environment, {name: os.environ.get(name) for name in names})
        for name in PROMPT_VARIABLES:
            os.environ.pop(name, None)
        os.environ[WELCOME_VARIABLE] = "true" if self.welcome else "false"

    def instrument(self, patches: ExitStack):
        """Count what the call has in flight, with wrappers that call the original methods unchanged."""
        case = self
        on_turn, speak, attach = CascadeSession.on_turn, LiveTalkerProcessor.speak, LiveTalkerProcessor.attach
        start = IMSession.start

        async def on_turn_counted(cascade):
            case.turns_started += 1
            case.turns_open += 1
            try:
                await on_turn(cascade)
            finally:
                case.turns_open -= 1
                case._changed()

        async def speak_counted(processor, text):
            case.speak_calls += 1
            await speak(processor, text)

        def attach_recorded(processor, session):
            case.sessions.append(session)
            attach(processor, session)

        def start_recorded(im):
            case.ims.append(im)
            start(im)

        patches.enter_context(patch.object(CascadeSession, "on_turn", on_turn_counted))
        patches.enter_context(patch.object(LiveTalkerProcessor, "speak", speak_counted))
        patches.enter_context(patch.object(LiveTalkerProcessor, "attach", attach_recorded))
        patches.enter_context(patch.object(IMSession, "start", start_recorded))

    def _changed(self):
        self._activity.set()

    # --------------------------------------------------------------------------------- the client's side
    async def client_ready(self):
        """The client reports ready, as the browser client does once connected; return the bot's ``bot-ready``.

        With the welcome on, the cascade then greets the caller with its own line (``lines.intro``), which the writer
        words; the greeting is never a caller turn.
        """
        after = len(self.transport.messages)
        await self.transport.receive(CLIENT_READY)
        return await self.wait_for_message("bot-ready", after=after)

    async def say(self, text):
        """The caller says ``text``; return the transcript message the client received for it."""
        after = len(self.transport.messages)
        self.lines_said += 1
        self.turns_expected += 1
        await self.stt.say(text)
        return await self.wait_for_message(
            "user-transcription", after=after, where=lambda m: m["data"]["text"].strip() == text.strip()
        )

    async def close(self):
        """The client disconnects; return once ``bot()`` has returned.

        It also checks how the bot ends: ``bot()`` returns without an error, the talker's model client is closed once
        and the backend's once if a backend round ran (the IM session's provider, made at its first round), nothing is
        spoken or asked of a model after the disconnect, and no task the bot started (its IM session, its silence timer,
        its pipeline) is still running once ``bot()`` has returned.
        """
        spoken, asked = len(self.tts.utterances), (len(self.frontend.requests), len(self.backend.requests))
        await self.transport.disconnect()
        done, _ = await asyncio.wait({self.bot}, timeout=WAIT_SECONDS)
        if not done:
            self.fail("bot() did not return after the client disconnected")
        if self.bot.exception() is not None:
            self.fail(f"bot() ended with an error: {self.bot.exception()!r}")
        closes = {client.role: client.closes for client in (self.frontend, self.backend)}
        expected = {"frontend": 1, "backend": 1 if self.backend.requests else 0}
        self.assertEqual(closes, expected, "each model client the call used is closed once when the bot ends")
        self.assertEqual(len(self.tts.utterances), spoken, "nothing is spoken after the disconnect")
        self.assertEqual((len(self.frontend.requests), len(self.backend.requests)), asked, "no model request after it")
        await self.transport.cleanup()  # the transport's own handler of the disconnect ends, as a transport's does
        left = sorted(task.get_name() for task in asyncio.all_tasks() - self.tasks_before)
        self.assertEqual(left, [], "no task the bot started is still running once bot() has returned")

    async def stop(self):
        """Clean-up for a test that did not close: end the bot as a disconnect does, else cancel it."""
        if self.bot.done():
            return
        await self.transport.disconnect()
        done, _ = await asyncio.wait({self.bot}, timeout=WAIT_SECONDS)
        if not done:
            self.bot.cancel()
            await asyncio.gather(self.bot, return_exceptions=True)

    # ------------------------------------------------------------------------------------------ waiting
    async def wait_for_message(self, kind, *, where=None, after=0, timeout=WAIT_SECONDS):
        """Wait for the first client message of type ``kind`` (at index ``after`` or later) that ``where`` accepts."""

        def found():
            return next(
                (
                    message
                    for message in self.transport.messages[after:]
                    if message.get("type") == kind and (where is None or where(message))
                ),
                None,
            )

        await self.wait_until(lambda: found() is not None, f"the client message {kind}", timeout)
        return found()

    def pending(self):
        """What the call still has in flight, as readable reasons; empty when it is quiet."""
        reasons = []
        if self.turns_started < self.turns_expected:
            reasons.append(f"{self.turns_expected - self.turns_started} turn(s) not yet at the talker")
        if self.turns_open:
            reasons.append(f"{self.turns_open} turn(s) being decided")
        if self.speak_calls > len(self.tts.utterances):
            reasons.append("speech on its way to the text-to-speech")
        return reasons + self.messages_pending() + self.bot_pending()

    def messages_pending(self):
        """Client messages the call has caused that the transport has not received yet.

        The messages per cause: per utterance a ``bot-llm-stopped`` (the processor's reply frames) and a
        ``bot-tts-stopped`` (the text-to-speech's); per caller line a ``user-transcription`` and the bot's
        ``user-turn-finalized`` server message (the task cards are server messages too, so it is counted by its own
        type); per turn the talker decides a ``user-llm-text``.
        """

        def kind(message):
            data = message.get("data")
            if message.get("type") == "server-message" and isinstance(data, dict):
                return f"server-message {data.get('type')}"
            return message.get("type")

        counts = Counter(kind(message) for message in self.transport.messages)
        expected = {
            "bot-llm-stopped": len(self.tts.utterances),
            "bot-tts-stopped": len(self.tts.utterances),
            "user-transcription": self.lines_said,
            "server-message user-turn-finalized": self.lines_said,
            "user-llm-text": self.turns_started,
        }
        return [
            f"{kind} messages not yet sent to the client ({counts[kind]} of {count})"
            for kind, count in expected.items()
            if counts[kind] < count
        ]

    def bot_pending(self):
        """The bot's delegated work in flight; a backend request the test holds is not. Re-point on a move.

        The cascade's presenter speaks each result in its own task, once the core has handed it over.
        """
        session = self.sessions[0] if self.sessions else None
        worker = self.ims[0] if self.ims else None
        reasons = []
        if worker is not None and worker.busy and not self.backend.parked:
            reasons.append("delegated work running")
        if session is not None and session.presenter.busy:
            reasons.append("results not yet spoken")
        return reasons

    # ------------------------------------------------------------------------------------- the golden
    def channels(self):
        """The call as recorded so far, by channel, before normalizing."""
        client = {"client_bot": [], "client_user": [], "client_server": []}
        for message in self.transport.messages:
            channel = client_channel(message)
            if channel is not None:
                client[channel].append(message)
        return {
            "spoken": list(self.tts.utterances),
            "frontend_requests": list(self.frontend.requests),
            "backend_requests": list(self.backend.requests),
            **client,
        }

    def assert_golden(self, name):
        """Compare the call with ``recordings/<name>.json`` channel by channel, or record it.

        As in the harness, only a run with ``LIVE_GOLDEN_RECORD=1`` writes the file, and it still fails when an
        existing file changed.
        """
        self.check_scripts_used()
        recorded = normalize_bot_channels(self.channels())
        path = GOLDEN_DIR / f"{name}.json"
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if os.environ.get(RECORD_VARIABLE) == "1":
            previous, golden = golden, {"name": name, "scenario": self.scenario(), "channels": recorded}
            GOLDEN_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            diffs = changed_bot_channels(name, previous.get("channels", {}), recorded) if previous else []
            if diffs:
                self.fail(
                    f"Recorded {path.name} again: {len(diffs)} channel(s) changed, and the file now holds this run. "
                    f"Check the diff, then run without {RECORD_VARIABLE}.\n\n" + "\n\n".join(diffs)
                )
        elif golden is None:
            self.fail(f"{path} does not exist; record it with {RECORD_VARIABLE}=1 (see the note in harness.py)")
        else:
            self.assertEqual(sorted(golden["channels"]), sorted(BOT_CHANNELS), f"{name}.json has the wrong channels")
            diffs = changed_bot_channels(name, golden["channels"], recorded)
            if diffs:
                self.fail(
                    f"Golden {name} changed in {len(diffs)} channel(s). If the change is intended, record again "
                    f"with {RECORD_VARIABLE}=1.\n\n" + "\n\n".join(diffs)
                )


# ----------------------------------------------------------------------------------------------- goldens
class BotGoldens(BotGoldenCase):
    """The repository UI bot's goldens."""

    async def test_x5_a_ui_bot_turn_with_a_cafe_server_tool(self):
        """X5: a repository UI-bot turn with a cafe server tool, through pipeline.bot().

        The client reports ready and the cascade greets the caller with its own line, which the writer words (the
        greeting is no caller turn). The caller asks for the teas; the talker delegates with a lead-in; the backend
        calls get_menu, which the bot's cafe tools run as a server tool, and its next round is held. Meanwhile the
        caller says thanks, and the talker, whose work note shows the delegation running, replies. The round is
        released, the writer speaks the backend's answer, and the client disconnects. Each delegation is a task card.
        """
        answer = Held(answer_round("We have a Matcha Latte for $5.00 and Earl Grey for $3.25."))
        await self.serve(
            frontend=[
                decision("speak", "Hi, I'm Remy at Bluebird Cafe. What can I get you?"),
                decision("delegate", "Let me check the menu."),
                decision("speak", "You're welcome, one moment."),
                decision("speak", "We have a Matcha Latte for five dollars and Earl Grey for three twenty-five."),
            ],
            backend=[tool_call_round(), answer],
        )
        await self.client_ready()
        await self.wait_quiet()
        await self.say("What teas do you have?")
        # Quiet: the lead-in is spoken, get_menu has run in the bot, and the round after it is held.
        await self.wait_quiet()
        await self.say("Thanks.")
        await self.wait_quiet()
        answer.release()
        await self.wait_quiet()
        await self.close()
        self.assert_golden("X5")

    async def test_x7_a_ui_bot_turn_whose_model_speaks_before_a_cafe_call(self):
        """X7: a UI-bot turn whose model says a sentence before a cafe call, through pipeline.bot().

        The welcome is off (ENABLE_WELCOME_MESSAGE=false), so the bot waits for the caller. The caller asks for the
        price of a large latte; the talker delegates with a lead-in. The backend's first round is held until the
        lead-in is spoken; it then says a sentence and calls get_menu in the same round, and the bot's cafe tools run
        the call. The second round answers, the writer speaks it, and the client disconnects. The sentence before the
        call is never spoken; the next round's input carries it.
        """
        first_round = Held(
            [
                chunk("I'll look up the latte "),
                chunk("on the menu first."),
                chunk(tool_calls=[tool_fragment(0, "call_1", "get_menu", '{"query":"latte"}')]),
                chunk(finish="tool_calls"),
            ]
        )
        await self.serve(
            frontend=[
                decision("delegate", "Let me look that up."),
                decision("speak", "A large latte is five fifty, before tax."),
            ],
            backend=[first_round, answer_round("A large Latte is $5.50 before tax.")],
            welcome=False,
        )
        await self.client_ready()
        await self.wait_quiet()
        await self.say("How much is a large latte?")
        # Quiet: the lead-in is spoken and written into the talker's context, and the first round is held.
        await self.wait_quiet()
        first_round.release()
        await self.wait_quiet()
        await self.close()
        self.assert_golden("X7")


class BotStartTests(BotGoldenCase):
    """What the bot's root starts: each loop runs once.

    The root opens its server delegation, whose ``open`` opens the conversation in the bot's core (its IM session's
    placer), and starts the cascade (its presenter and the silence timer). Attaching the cascade to its processor
    starts neither.
    """

    @staticmethod
    def running(name):
        return [task for task in asyncio.all_tasks() if task.get_name() == name and not task.done()]

    async def test_the_conversations_im_session_starts_once(self):
        await self.serve(welcome=False)
        self.assertEqual(len(self.ims), 1)
        self.assertEqual(len(self.running("im-placer")), 1)
        await self.close()

    async def test_the_silence_timer_starts_once(self):
        await self.serve(welcome=False)
        self.assertEqual(len(self.running("delegation-progress")), 1)
        await self.close()


class BotCloseTests(BotGoldenCase):
    """What the bot's root closes: its cascade, then its server delegation, then its core, once each.

    The talker processor closes nothing, so the root closes the cascade (the presenter's tasks, the
    silence timer and the talker's model) once its pipeline has finished, or in its ``finally`` at the latest, before
    the server delegation, whose close ends the conversation's IM session (and the provider of a backend round). A
    client's disconnect ends ``bot()`` normally, so :meth:`BotGoldenCase.close` cannot tell a ``finally`` from a plain
    last line; a run that fails can.
    """

    class FailingRunner:
        """A pipeline runner that fails before the pipeline starts."""

        def __init__(self, *args, **kwargs):
            pass

        async def add_workers(self, *workers):
            raise RuntimeError("the runner failed")

    async def serve_failing(self):
        """``serve`` with a runner that fails before the pipeline starts; return the bot's cascade."""
        with (
            patch.object(pipeline, "WorkerRunner", self.FailingRunner),
            self.assertRaisesRegex(AssertionError, "ended before its pipeline started"),
        ):
            await self.serve(welcome=False)
        [session] = self.sessions
        return session

    def assert_conversation_closed(self):
        """The conversation's IM session closed, and no backend round ran, so no backend model client was made."""
        [im] = self.ims
        self.assertTrue(im.closed and im.placer.done(), "the conversation's IM session still runs")
        self.assertEqual(self.backend.closes, 0, "no backend round ran, so no backend model client was made")

    async def test_the_talkers_model_is_released_before_the_backend_model(self):
        # One delegation, so the backend's IM session has made the provider of its round.
        await self.serve(
            frontend=[decision("delegate", "Let me check."), decision("speak", "We have tea.")],
            backend=[answer_round("We have tea.")],
            welcome=False,
        )
        await self.client_ready()
        await self.wait_quiet()
        await self.say("What do you have?")
        await self.wait_quiet()
        released = []
        for client in (self.frontend, self.backend):

            async def noted_close(close=client.close, role=client.role):
                released.append(role)
                await close()

            client.close = noted_close
        await self.close()
        self.assertEqual(released, ["frontend", "backend"])

    async def test_a_bot_whose_run_fails_still_closes_its_cascade(self):
        await self.serve_failing()
        self.assertEqual(str(self.bot.exception()), "the runner failed")
        self.assertEqual(self.frontend.closes, 1, "the talker's model is released once")
        self.assert_conversation_closed()
        self.assertEqual(BotStartTests.running("delegation-progress"), [], "the silence timer still runs")

    async def test_a_cascade_that_cannot_close_still_lets_the_delegation_side_close(self):
        cascade_close = CascadeSession.close

        async def failing_close(session):
            raise RuntimeError("the cascade could not close")

        patcher = patch.object(CascadeSession, "close", failing_close)
        patcher.start()
        self.addCleanup(patcher.stop)
        session = await self.serve_failing()
        # The cascade's failure is logged; bot() still raises the runner's own error.
        self.assertEqual(str(self.bot.exception()), "the runner failed")
        self.assert_conversation_closed()
        await cascade_close(session)  # the silence timer that the failed close left running

    async def test_a_bot_cancelled_while_its_cascade_closes_still_closes_its_delegation_side(self):
        cascade_close, closing = CascadeSession.close, asyncio.Event()

        async def hung_close(session):
            closing.set()
            await asyncio.Event().wait()  # the bot is cancelled here

        patcher = patch.object(CascadeSession, "close", hung_close)
        patcher.start()
        self.addCleanup(patcher.stop)

        async def cancel_the_bot():
            await closing.wait()
            self.bot.cancel()

        canceller = asyncio.create_task(cancel_the_bot())
        # serve() reads the ended bot's exception, which a cancelled task raises.
        with patch.object(pipeline, "WorkerRunner", self.FailingRunner), self.assertRaises(asyncio.CancelledError):
            await self.serve(welcome=False)
        await canceller
        [session] = self.sessions
        self.assertTrue(self.bot.cancelled())
        self.assert_conversation_closed()
        await cascade_close(session)  # the silence timer that the cut close left running


class BotWiringTests(BotGoldenCase):
    """What ``bot()`` builds: its processors in order, the cafe tools as the backend's server tools, and its buffer.

    This is the bot wiring test: X5 and X7 pin what the bot says and asks, and this test
    pins how it is put together (``pipeline.bot``).
    """

    def replace_edges(self, patches):
        """The goldens' edges, plus a note of the processors ``bot()`` hands to its ``Pipeline``."""
        super().replace_edges(patches)
        self.processors: list[list] = []
        build = pipeline.Pipeline

        def noted_pipeline(processors, *args, **kwargs):
            self.processors.append(list(processors))
            return build(processors, *args, **kwargs)

        patches.enter_context(patch.object(pipeline, "Pipeline", noted_pipeline))

    async def test_the_bot_wires_the_cascade_the_cafe_tools_and_the_talkers_trim(self):
        await self.serve(frontend=[decision("speak", "Hello there.")], welcome=False)
        await self.client_ready()
        await self.wait_quiet()

        # The processors in order (``bot`` in pipeline.py): the transport's input, speech to text, the user
        # aggregator, the talker processor, text to speech (with its mode's text hold, none for this one) and the
        # transport's output, the playout tracker, then the audio recorder when a dump is switched on. No assistant
        # aggregator: the cascade writes its own lines into the context, and the tracker corrects them by playout.
        [processors] = self.processors
        session = self.sessions[0]
        kinds = [
            FrameProcessor,
            FakeSTT,
            LLMUserAggregator,
            LiveTalkerProcessor,
            RecordingTTS,
            FrameProcessor,
            PlayoutTracker,
        ]
        self.assertEqual(len(processors), len(kinds), processors)
        for processor, kind in zip(processors, kinds, strict=True):
            self.assertIsInstance(processor, kind)
        self.assertEqual(
            (processors[0], processors[1], processors[3].session, processors[4], processors[5]),
            (self.transport.input(), self.stt, session, self.tts, self.transport.output()),
        )
        # The backend runs the cafe tools itself, as server tools: the conversation's IM session's backend adapter holds
        # the tool sources core.ui-bot.yaml names (backend.tools: cafe), with their schemas; the core's defaults offer
        # no request tools of their own.
        im = self.ims[0]
        self.assertEqual([type(source) for source in im.backend.tools], [CafeTools])
        self.assertEqual(im.backend.tools[0].tools(), CAFE_TOOLS)
        self.assertEqual(im.defaults.tools, [])

        # The bot's Pipecat context is only the host's buffer of caller turns (TurnBuffer): the cascade takes each
        # committed turn into the talker's memory and trims the buffer back to its prompt, and nothing else writes to
        # it (no assistant aggregator: the cascade writes its own lines). The talker's memory holds the turn and
        # the reply.
        buffer = session.buffer.context
        prompt = list(buffer.get_messages())  # a copy: the context may change its own list in place
        self.assertEqual({message["role"] for message in prompt}, {"system"})
        await self.say("Hi.")
        await self.wait_quiet()

        def replied():
            return session.memory.messages()[-1]["role"] == "assistant"

        await self.wait_until(replied, "the reply written into the talker's memory")
        self.assertEqual(buffer.get_messages(), prompt)
        self.assertEqual(
            [(message["role"], message["content"]) for message in session.memory.messages()[-2:]],
            [("user", "Hi."), ("assistant", "Hello there.")],
        )
        self.check_scripts_used()
        await self.close()
