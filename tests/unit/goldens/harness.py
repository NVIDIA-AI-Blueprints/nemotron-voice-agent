# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Golden transcripts: whole calls through the real gateway, the real cascade adapter and the real core, by channel.

A golden serves one call on the real ``LiveGateway`` (its primary WebSocket loop) with the parts the root builds: the
cascade adapter, and the gateway's one delegation core from the shipped ``core.yaml`` (``build_core``, its backend LLM
on the backend's script). Speech is faked (``FakeSTT``, ``FakeTTS``) and each model role reads its own script, so the
call runs the same way every time. The harness records the call in seven channels, normalizes them, and compares each
channel on its own with ``recordings/<name>.json``:

* ``session``: the session-level server events (no audio, transcript or delegation events);
* ``delegations``: the events of each delegation in order (``session.delegation.created`` and ``response.event``),
  keyed by the delegation id;
* ``input_transcript`` and ``output_transcript``: each transcript event, written as its ``delta`` text when it has
  exactly these keys (``type``, ``event_id``, ``delta``, ``start_ms``, ``end_ms``) and whole otherwise, so a change
  to its shape shows;
* ``spoken``: each utterance the text-to-speech received, in the order it spoke them;
* ``frontend_requests`` and ``backend_requests``: the arguments each scripted model client received. The frontend
  client serves the talker and the writer; the backend client serves the backend rounds.

Each channel is compared alone because transcript events are sent from separate tasks
(``examples/frontend_backend_live/adapter/bridge.py``), so their order relative to other events is not fixed. Audio
packets (``session.output_audio.delta``) are in no channel: the gateway sends one every 100 ms of wall-clock time,
silence included, so their number differs from run to run (``tests/unit/cross_box/test_live_pipeline.py`` pins the audio
stream). :meth:`GoldenCase.check_event_ids` still checks that none carries an ``event_id``.

The normalizer replaces each generated id (``<prefix>_<32 hex digits>``) with a placeholder numbered in order of first
use, per prefix and per channel (``<item_1>``, ``<resp_2>``); the delegations share one numbering, so a response id that
a second delegation reused would show. Server ``event_id`` values all become ``<event>``
(:meth:`GoldenCase.check_event_ids` checks that no two events share one), and clock values become ``<clock>`` with
their key kept. A string with line breaks is stored as the list of its lines, so a prompt change shows as one changed
line. Channels are compared as JSON text with sorted keys: ``true`` and ``1`` differ, and so do ``1`` and ``1.0``.

Usage
-----

**Script a call.** Subclass :class:`GoldenCase` and write one test per golden::

    class MenuGoldens(GoldenCase):
        async def test_g17_a_delegation_with_one_client_tool_round(self):
            '''G17: one or two sentences on what the call does; they are copied into the golden file.'''
            await self.serve(
                frontend=[decision("delegate", "Let me check."), decision("speak", "We have tea.")],
                backend=[tool_call_round(), answer_round("We have tea.")],
            )
            await self.say("What teas do you have?")
            call = await self.wait_for_call("get_menu")
            await self.wait_quiet()
            self.send(result_item(call["call_id"], "{}"))
            self.send({"type": "response.create", "event_id": "continue"})
            await self.wait_quiet()
            await self.close()
            self.assert_golden("G17")

* ``serve(session=None, *, frontend=(), backend=(), live_config=None, delegation=None)`` starts the session and returns
  ``session.started``; ``delegation`` sets keys of the core's ``delegation`` block (``core.yaml``). ``frontend`` lists
  the talker's and the writer's replies in the order they are asked for; ``backend`` lists the backend rounds in order.
  A step is a chunk list or an exception, as ``FakeChat`` takes it: ``decision(action, speech)``, ``tool_call_round()``,
  ``answer_round(text)``, ``[chunk(...)]``. Every step must be used and no client may run out of steps; ``wait_quiet``
  and ``assert_golden`` fail otherwise. ``session`` defaults to ``RESPONSES_SESSION`` and ``live_config`` to
  ``GOLDEN_LIVE_CONFIG``.
* ``say(text)`` speaks a caller line and returns its input transcript event (pass ``turn=False`` for a line that will
  not reach the talker); ``feed_silence(tenths)`` sends caller audio; ``send(event)`` sends any client event.
* ``wait_for(kind, where=...)``, ``wait_for_response_event(kind, where=...)`` and ``wait_for_call(name)`` wait for one
  server event; ``wait_until(predicate, what)`` waits for any condition; ``wait_quiet()`` waits until nothing is in
  flight (see :meth:`GoldenCase.pending`), bounded. Wait for an event or for quiet before each client action, so the
  call runs one way only; never sleep. ``wait_quiet`` does not wait for append acknowledgments, which wait for the
  input clock: feed silence and wait for the ``...appended`` event. Speech cut by an interruption never reaches the
  text-to-speech, so a barge-in golden waits for events rather than for quiet.
* ``close()`` closes the session as the client does and checks that ``session.closed`` came once and last;
  ``assert_golden(name)`` then compares every channel.
* Timers. ``GOLDEN_LIVE_CONFIG`` sets the quiet before a progress line to an hour (``progress_after_seconds``),
  and the gateway's usage report is patched to an hour (``usage_interval_seconds``), so neither puts a clock-driven
  line into a golden. A golden that tests a timer sets it small through its own ``live_config`` or the core's
  ``delegation`` keys, or a test-local patch, with a comment, for example ``replace(GOLDEN_LIVE_CONFIG,
  delegation=replace(GOLDEN_LIVE_CONFIG.delegation, progress_after_seconds=0.2))`` or
  ``delegation={"client_tool_timeout_seconds": 0.2}``.
* Held rounds. A :class:`HeldRound` backend step sends its request and relays its opening events, then waits until the
  test releases it, as a slow model does. A golden that holds rounds builds its clients as :class:`HoldingChat` (patch
  ``harness.ScriptedChat`` while ``serve`` runs) and drops the reason :data:`WORK_RUNNING` from ``engine_pending``
  while ``self.backend.holding``: a held round waits for the test, as a client-tool round waits for the client.

**Record.** ``LIVE_GOLDEN_RECORD=1 uv run pytest <test file> -q -p no:cacheprovider`` writes
``recordings/<name>.json``: the scenario (the test's docstring) and the channels. A recording that changes an existing
file still writes it, then fails with the diff, so a record run never passes over a change unseen; a new file or an
unchanged one passes. A normal run only compares, never writes, and fails with a unified diff per changed channel.
After recording, read the file, then run the test several times in a row. The folder is not called ``data`` because
the repository's ``.gitignore`` ignores every folder of that name. The goldens import this module as
``tests.unit.goldens.harness``, which needs pytest's ``pythonpath = ["src", "."]`` (``pyproject.toml``).

**Internals.** Two methods are the only places that reach past the wire: :meth:`GoldenCase.instrument` wraps five
production methods (``LiveProtocolSession.handle``, ``CascadeSession.on_turn``, ``LiveTalkerProcessor.speak``,
``TaskRun._wait_for_app`` and ``WorkNote.quiet_candidate``) with counting wrappers that call the original unchanged, and
:meth:`GoldenCase.engine_pending` reads the cascade adapter's, the core's and the gateway's state. When a change moves
those objects, re-point these two methods; the goldens themselves stay. A spoken line enters the talker's
context when it is handed to speech, so nothing waits for it to be written there.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import inspect
import json
import os
import re
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pipecat.frames.frames import LLMTextFrame

from delegation_core.im.run import TaskRun
from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.worknote import WorkNote
from examples.frontend_backend_live.media.processor import LiveTalkerProcessor
from examples.frontend_backend_live.prompts import get_prompt_set
from live.gateway import LiveGateway, LiveSettings
from live.gateway.websocket import run_primary_websocket
from live.ports import SessionParts
from live.session import LiveProtocolSession, lifecycle
from tests.unit.cascade_helpers import (
    ENDPOINT,
    LIVE_CONFIG,
    MENU_TOOL,
    STATIC_BACKEND,
    FakeSTT,
    FakeTTS,
    _decision_chunks,
    cascade_parts,
    chat_core,
    core_hook,
    result_item,
    tool_call_round,
)
from tests.unit.core_helpers import FakeChat, chunk, tool_fragment
from tests.unit.cross_box.test_frontend_backend_live_engine import RESPONSES_SESSION, EngineHarness
from tests.unit.live_helpers import FakeWebSocket

__all__ = [
    "CHANNELS",
    "ENDPOINT",
    "GOLDEN_LIVE_CONFIG",
    "MENU_TOOL",
    "RESPONSES_SESSION",
    "RESULTS_PENDING",
    "STATIC_BACKEND",
    "WORK_RUNNING",
    "GoldenCase",
    "HeldRound",
    "HoldingChat",
    "Normalizer",
    "answer_round",
    "chunk",
    "decision",
    "normalize_channels",
    "result_item",
    "tool_call_round",
    "tool_fragment",
]

# Not "data": the repository's .gitignore ignores every folder of that name, so the goldens would never be committed.
GOLDEN_DIR = Path(__file__).parent / "recordings"
RECORD_VARIABLE = "LIVE_GOLDEN_RECORD"

# Timers that would put clock-driven lines into a golden. A golden that tests one of them sets it small itself.
# A quiet spell's progress line may come after 8 s of silent work by default; an hour keeps it out.
QUIET_PROGRESS_SECONDS = 3600.0
# The gateway reports session.usage.updated every 5 s (live.session.lifecycle.USAGE_INTERVAL_SECONDS); the harness
# patches the interval to an hour for the length of each test.
USAGE_INTERVAL_SECONDS = 3600
# The talker's prompt set the goldens run on: the shipped one, so a change to its texts shows in every golden's frontend
# requests.
GOLDEN_PROMPT_SET = "default"
GOLDEN_PROMPTS = get_prompt_set(GOLDEN_PROMPT_SET)
# The core's voice note the goldens run on: the shipped one, which every golden's backend requests carry.
GOLDEN_CORE = {"voice_note": "default"}
# The models the core's catalog lists besides the session's: a session.update may choose them (core.yaml reports an
# unlisted one).
GOLDEN_MODELS = ("org/other-model",)
GOLDEN_LIVE_CONFIG = replace(
    LIVE_CONFIG,
    prompt_version=GOLDEN_PROMPT_SET,
    delegation=replace(LIVE_CONFIG.delegation, progress_after_seconds=QUIET_PROGRESS_SECONDS),
)

WAIT_SECONDS = 10.0
# How often a wait looks again at state that announces no change of its own (an IM session going idle).
POLL_SECONDS = 0.01
# The reason GoldenCase.engine_pending gives while the IM session is busy and not waiting for the client's
# tool outputs. A golden that holds backend work drops it while the work is held; import it, never copy the text.
WORK_RUNNING = "delegated work running"
# The reason it gives while the cascade's presenter has a result it has not spoken yet. A golden
# drops it only while the test holds the writer's reply to a result; import it, never copy the text.
RESULTS_PENDING = "results not yet spoken"
# Loop turns given to callbacks that are already scheduled before a quiet call counts as quiet.
SETTLE_STEPS = 20

CHANNELS = (
    "session",
    "delegations",
    "input_transcript",
    "output_transcript",
    "spoken",
    "frontend_requests",
    "backend_requests",
)


# --------------------------------------------------------------------------------------------- script steps
def decision(action, speech=""):
    """A talker or writer step: the ``{action, speech}`` reply, streamed in chunks as ``_decision_chunks`` does."""
    return _decision_chunks(json.dumps({"action": action, "speech": speech}, separators=(",", ":"), ensure_ascii=False))


def answer_round(text):
    """A backend round that answers with ``text`` and calls no tool."""
    return [chunk(text, finish="stop")]


# ------------------------------------------------------------------------------------------- the normalizer
GENERATED_ID = re.compile(r"\b([a-z]+)_([0-9a-f]{32})\b")
# Ids that are fresh on every event: numbering them would only restate each event's position in its channel. That
# every event still has its own is checked by GoldenCase.check_event_ids.
UNNUMBERED_PREFIXES = frozenset({"event"})
FRESH_EVENT_ID = re.compile(r"event_[0-9a-f]{32}")
# Keys whose number or text is a wall-clock or media-clock reading. The value becomes CLOCK and the key stays, so a
# time field that appears or goes away still shows in the diff.
CLOCK_KEYS = frozenset({"created_at", "expires_at", "offset_ms", "start_ms", "end_ms", "timestamp"})
CLOCK = "<clock>"
# The work note's running state counts the seconds the work has run, in the golden set's words: a clock reading too.
RUNNING = GOLDEN_PROMPTS.work_texts.running
CLOCK_TEXT = (
    (
        re.compile(r"\b" + r"\d+".join(map(re.escape, RUNNING.split("{seconds}"))) + r"\b"),
        RUNNING.format(seconds="<seconds>"),
    ),
)
AUDIO_EVENT = "session.output_audio.delta"
# A transcript event with exactly these keys is written as its delta text; any other is written whole.
TRANSCRIPT_KEYS = frozenset({"type", "event_id", "delta", "start_ms", "end_ms"})
TRANSCRIPT_KINDS = {
    "input_transcript": "session.input_transcript.delta",
    "output_transcript": "session.output_transcript.delta",
}


class Normalizer:
    """Normalizes one channel: ids become first-seen placeholders per prefix; clock values become ``<clock>``."""

    def __init__(self):
        self.ids: dict[str, str] = {}
        self.counts: dict[str, int] = {}

    def __call__(self, value, key=None):
        if isinstance(value, dict):
            return {
                self.ids_in(name): CLOCK if self.is_clock(key, name, item) else self(item, name)
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self(item, key) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    @staticmethod
    def is_clock(parent, key, value):
        """A clock reading: a number or a text under a clock key. Objects, lists, null and booleans are not."""
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return False
        return key in CLOCK_KEYS or (parent == "usage" and key == "seconds")

    def ids_in(self, value):
        """``value`` with each generated id replaced (a key that is not a string is kept as it is)."""
        return GENERATED_ID.sub(self._placeholder, value) if isinstance(value, str) else value

    def text(self, value):
        value = self.ids_in(value)
        for pattern, replacement in CLOCK_TEXT:
            value = pattern.sub(replacement, value)
        lines = value.split("\n")
        return lines if len(lines) > 1 else value

    def _placeholder(self, match):
        raw, prefix = match.group(0), match.group(1)
        if prefix in UNNUMBERED_PREFIXES:
            return f"<{prefix}>"
        if raw not in self.ids:
            self.counts[prefix] = self.counts.get(prefix, 0) + 1
            self.ids[raw] = f"<{prefix}_{self.counts[prefix]}>"
        return self.ids[raw]


def transcript_line(event, kind):
    """A normalized transcript event as its ``delta`` text when it has exactly the plain shape; otherwise the event."""
    if (
        set(event) == TRANSCRIPT_KEYS
        and event["type"] == kind
        and event["event_id"] == "<event>"
        and event["start_ms"] == event["end_ms"] == CLOCK
        and isinstance(event["delta"], str)
    ):
        return event["delta"]
    return event


def normalize_channels(raw):
    """Normalize every channel on its own; the delegations share one normalizer, taken in order of first event."""
    normalized = {}
    for channel in CHANNELS:
        normalize = Normalizer()
        if channel == "delegations":
            normalized[channel] = {
                normalize.ids_in(delegation_id): normalize(events) for delegation_id, events in raw[channel].items()
            }
        elif channel in TRANSCRIPT_KINDS:
            kind = TRANSCRIPT_KINDS[channel]
            normalized[channel] = [transcript_line(normalize(event), kind) for event in raw[channel]]
        else:
            normalized[channel] = normalize(raw[channel])
    return normalized


# ---------------------------------------------------------------------------------------------------- diffs
def render(value):
    return json.dumps(value, indent=2, ensure_ascii=False).splitlines()


def canonical(value):
    """The JSON text a channel is compared by: ``true`` and ``1`` differ, the order of keys does not."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def channel_diff(name, channel, expected, recorded):
    lines = difflib.unified_diff(
        render(expected), render(recorded), f"{name}.json: {channel} (golden)", f"{channel} (this run)", lineterm=""
    )
    return "\n".join(lines)


def changed_channels(name, expected, recorded):
    """A unified diff for each channel whose JSON differs from ``expected`` (a golden file's channels)."""
    return [
        channel_diff(name, channel, expected.get(channel), recorded[channel])
        for channel in CHANNELS
        if canonical(expected.get(channel)) != canonical(recorded[channel])
    ]


# ------------------------------------------------------------------------------------------------- doubles
class ScriptExhausted(Exception):
    """A model client was called after its script ran out."""


class ScriptedChat(FakeChat):
    """``FakeChat`` for one model role: steps are used in order, and each request is kept exactly as it was sent."""

    def __init__(self, role, *script, on_request=None):
        super().__init__(*script)
        self.role = role
        self.on_request = on_request or (lambda: None)
        self.overrun: list[int] = []  # numbers of the requests that found the script empty

    async def create(self, **kwargs):
        sent = json.loads(json.dumps(kwargs, default=repr))
        if not self.script:
            self.requests.append(sent)
            self.overrun.append(len(self.requests))
            self.on_request()
            raise ScriptExhausted(f"the {self.role} script has no step for request {len(self.requests)}")
        try:
            return await super().create(**kwargs)
        finally:
            self.requests[-1] = sent  # a copy, so a later change to the caller's lists cannot rewrite the record
            self.on_request()


class HeldRound(list):
    """A backend step whose model answers only when the test releases it.

    The round's request is sent and its delegation's opening events are relayed (the core writes ``response.created``
    and ``response.in_progress`` when the work starts, before the model is called); its chunks stream after
    :meth:`release`. A round that is never released waits until it is cancelled, like a model that has not answered
    when the call closes; ``on_cancel`` is called then.

    Holding a round lets the call settle first (the lead-in spoken and written into the talker's context), so the
    requests that follow the round see the same conversation on every run.
    """

    def __init__(self, chunks=()):
        super().__init__(chunks)
        self.waiting = asyncio.Event()  # the model was asked and has not answered
        self.released = asyncio.Event()
        self.cancelled = self.closed = False
        self.on_cancel = lambda: None

    def release(self):
        self.released.set()


class HeldStream:
    """The stream of a held round: it waits for the release before its first chunk and records how it ended."""

    def __init__(self, held: HeldRound, on_change):
        self.held, self.on_change = held, on_change

    def __aiter__(self):
        """Iterate the round's chunks once the test releases it."""
        return self._iterate()

    async def _iterate(self):
        self.held.waiting.set()
        self.on_change()
        try:
            await self.held.released.wait()
        except asyncio.CancelledError:
            self.held.cancelled = True
            self.held.on_cancel()
            raise
        for item in self.held:
            yield item

    async def close(self):
        self.held.closed = True


class HoldingChat(ScriptedChat):
    """``ScriptedChat`` in which a :class:`HeldRound` step streams only once the test releases it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held: list[HeldRound] = []

    async def create(self, **kwargs):
        stream = await super().create(**kwargs)
        if isinstance(stream.chunks, HeldRound):
            self.held.append(stream.chunks)
            return HeldStream(stream.chunks, self.on_request)
        return stream

    @property
    def holding(self):
        """Whether a held round waits for the test now."""
        return any(held.waiting.is_set() and not held.released.is_set() for held in self.held)


class RecordingTTS(FakeTTS):
    """``FakeTTS`` that also keeps each utterance it was given, in the order it spoke them."""

    def __init__(self, on_change=None, **kwargs):
        super().__init__(**kwargs)
        self.on_change = on_change or (lambda: None)
        self.utterances: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.utterances.append(frame.text)
            self.on_change()


class RecordingWebSocket(FakeWebSocket):
    """``FakeWebSocket`` that reports each server event and counts the client commands sent after the start."""

    def __init__(self, on_event):
        super().__init__()
        self.on_event = on_event
        self.commands = 0

    def send(self, message):
        if message.get("type") != "session.start":
            self.commands += 1
        super().send(message)

    async def send_json(self, message):
        await super().send_json(message)
        self.on_event()


# ---------------------------------------------------------------------------------------------- the harness
class GoldenCase(EngineHarness):
    """One scripted call per test on the real gateway, adapter and core, pinned to ``recordings/<name>.json``."""

    session = RESPONSES_SESSION
    live_config = GOLDEN_LIVE_CONFIG
    usage_interval_seconds = USAGE_INTERVAL_SECONDS

    # ---------------------------------------------------------------------------------------- starting
    async def serve(self, session=None, *, frontend=(), backend=(), live_config=None, delegation=None):
        """Start a session on the real gateway, adapter and core, and return ``session.started``.

        Unlike ``EngineHarness.serve``, the frontend (talker and writer) and the backend each get their own scripted
        client, so each script is consumed in a fixed order whatever the two roles do at the same time. ``delegation``
        sets keys of the core's ``delegation`` block.
        """
        self.frontend = ScriptedChat("frontend", *frontend, on_request=self._changed)
        self.backend = ScriptedChat("backend", *backend, on_request=self._changed)
        self.stt, self.tts = FakeSTT(), RecordingTTS(on_change=self._changed)
        self.engines: list[CascadeAdapter] = []
        self.parts: list[SessionParts] = []
        self.handled = self.turns_expected = self.turns_started = self.turns_open = self.speak_calls = 0
        self.tool_waits: set[TaskRun] = set()  # the IM's tasks in their wait for the client's tool outputs
        self.quiet_looks: list[
            bool
        ] = []  # each look of the quiet-spell timer: whether it found a delegation to speak of
        self._activity = asyncio.Event()
        patches = ExitStack()
        self.addCleanup(patches.close)  # cleanups run last-in first-out: the session closes before the patches go
        self.instrument(patches)
        live_config = live_config or self.live_config

        async def factory(config, transport, connection):
            parts = cascade_parts(
                config, transport, connection, chat=self.frontend, stt=self.stt, tts=self.tts, live_config=live_config
            )
            self.engines.append(parts.frontend)
            self.parts.append(parts)
            return parts

        # The gateway's one core, as the root builds it from core.yaml: its backend LLM reads the backend's script.
        self.core = chat_core(self.backend, models=GOLDEN_MODELS, **{**GOLDEN_CORE, **(delegation or {})})
        self.addAsyncCleanup(self.core.aclose)
        hook = core_hook(self.core)
        self.core_client = hook.client
        self.gateway = LiveGateway(factory, LiveSettings(max_session_seconds=60), core=hook)
        self.ws = RecordingWebSocket(self._changed)
        self.server = asyncio.create_task(run_primary_websocket(self.gateway, self.ws))
        self.addAsyncCleanup(self.stop)
        self.ws.send({"type": "session.start", "event_id": "start", "session": session or self.session})
        return await self.wait_for("session.started")

    @property
    def engine(self) -> CascadeAdapter:
        return self.engines[0]

    def instrument(self, patches: ExitStack):
        """Count what the call has in flight, with wrappers that call the original methods unchanged."""
        case = self
        handle, on_turn, speak = LiveProtocolSession.handle, CascadeSession.on_turn, LiveTalkerProcessor.speak
        wait_for_app, quiet_candidate = TaskRun._wait_for_app, WorkNote.quiet_candidate

        async def handle_counted(session, message, connection):
            try:
                await handle(session, message, connection)
            finally:
                case.handled += 1
                case._changed()

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

        async def wait_for_app_counted(task):
            case.tool_waits.add(task)
            case._changed()
            try:
                return await wait_for_app(task)
            finally:
                case.tool_waits.discard(task)
                case._changed()

        def quiet_candidate_counted(note, max_lines):
            entry = quiet_candidate(note, max_lines)
            case.quiet_looks.append(entry is not None)
            case._changed()
            return entry

        patches.enter_context(patch.object(LiveProtocolSession, "handle", handle_counted))
        patches.enter_context(patch.object(WorkNote, "quiet_candidate", quiet_candidate_counted))
        patches.enter_context(patch.object(CascadeSession, "on_turn", on_turn_counted))
        patches.enter_context(patch.object(LiveTalkerProcessor, "speak", speak_counted))
        patches.enter_context(patch.object(TaskRun, "_wait_for_app", wait_for_app_counted))
        # The usage report is a clock: an hour keeps it out of the session channel (see USAGE_INTERVAL_SECONDS).
        patches.enter_context(patch.object(lifecycle, "USAGE_INTERVAL_SECONDS", self.usage_interval_seconds))

    def _changed(self):
        self._activity.set()

    # --------------------------------------------------------------------------------- the client's side
    async def say(self, text, *, turn=True):
        """The caller says ``text``; return its input transcript event.

        ``turn=False`` is for a line that will not reach the talker, so ``wait_quiet`` does not wait for its turn.
        """
        after = len(self.ws.from_server)
        if turn:
            self.turns_expected += 1
        await self.stt.say(text)
        return await self.wait_for(
            "session.input_transcript.delta", after=after, where=lambda e: e["delta"].strip() == text.strip()
        )

    def send(self, message):
        """Send any client event on the primary WebSocket."""
        self.ws.send(message)

    async def close(self, event_id="close"):
        """Close the session as the client does; return ``session.closed`` once the gateway loop has ended.

        It also checks the close order: ``session.closed`` comes once and last. The channels are compared one by
        one, so this is where an event that moved past ``session.closed`` would show.
        """
        after = len(self.ws.from_server)
        self.send({"type": "session.close", "event_id": event_id})
        closed = await self.wait_for("session.closed", after=after)
        try:
            async with asyncio.timeout(WAIT_SECONDS):
                await self.server
        except TimeoutError:
            self.fail("The gateway loop did not end after session.closed")
        kinds = [event["type"] for event in self.ws.from_server]
        later = kinds[kinds.index("session.closed") + 1 :]
        if kinds.count("session.closed") != 1 or later:
            self.fail(
                f"session.closed must come once and last; the client received {kinds.count('session.closed')}, "
                f"then {later}"
            )
        return closed

    # ------------------------------------------------------------------------------------------ waiting
    async def wait_for(self, kind, *, where=None, after=0, timeout=WAIT_SECONDS, what=None):
        """Wait for the first server event of type ``kind`` (at index ``after`` or later) that ``where`` accepts."""
        try:
            return await self.ws.wait_for(kind, after=after, timeout=timeout, where=where)
        except TimeoutError:
            seen = [e["type"] for e in self.ws.from_server[after:] if e["type"] != AUDIO_EVENT]
            self.fail(f"No {what or kind} within {timeout} s; from event {after} on the client received {seen}")

    async def wait_for_response_event(self, kind, *, where=None, after=0, timeout=WAIT_SECONDS):
        """Wait for a relayed backend event of type ``kind``; ``where`` sees the inner event."""
        return await self.wait_for(
            "response.event",
            after=after,
            timeout=timeout,
            where=lambda e: e["event"]["type"] == kind and (where is None or where(e["event"])),
            what=f"response.event {kind}",
        )

    async def wait_for_call(self, name=None, *, after=0, timeout=WAIT_SECONDS):
        """Wait for a completed function call relayed to the client and return its item."""

        def is_call(event):
            item = event.get("item", {})
            return item.get("type") == "function_call" and (name is None or item.get("name") == name)

        relayed = await self.wait_for_response_event(
            "response.output_item.done", where=is_call, after=after, timeout=timeout
        )
        return relayed["event"]["item"]

    async def wait_until(self, predicate, what, timeout=WAIT_SECONDS):
        """Wait until ``predicate()`` is true; it is checked on every recorded change and every ``POLL_SECONDS``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._activity.clear()
            if predicate():
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.fail(f"Timed out after {timeout} s waiting for {what}")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._activity.wait(), min(POLL_SECONDS, remaining))

    async def wait_quiet_spells(self, looks):
        """Wait for ``looks`` more looks of the quiet-spell timer; return whether any found work it could still say.

        The timer looks only while the floor is idle and no result waits, so counting its looks counts quiet time
        without a sleep: a golden that checks a quiet spell's cap waits for enough looks to span the spell.
        """
        start = len(self.quiet_looks)
        await self.wait_until(lambda: len(self.quiet_looks) >= start + looks, f"{looks} looks of the quiet-spell timer")
        return any(self.quiet_looks[start:])

    async def wait_quiet(self, timeout=WAIT_SECONDS):
        """Wait until the call has nothing in flight (see :meth:`pending`): the server waits for the client.

        Quiet must hold again after ``SETTLE_STEPS`` loop turns, so callbacks already scheduled have run.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._activity.clear()
            self.check_scripts_not_overrun()
            pending = self.pending()
            if not pending:
                for _ in range(SETTLE_STEPS):
                    await asyncio.sleep(0)
                if not self.pending():
                    return
                continue
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.fail(f"The call did not go quiet within {timeout} s: {'; '.join(pending)}")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._activity.wait(), min(POLL_SECONDS, remaining))

    def pending(self):
        """What the call still has in flight, as readable reasons; empty when it is quiet."""
        reasons = []
        if self.ws.commands > self.handled:
            reasons.append(f"{self.ws.commands - self.handled} client command(s) not handled yet")
        if self.turns_started < self.turns_expected:
            reasons.append(f"{self.turns_expected - self.turns_started} caller line(s) not yet at the talker")
        if self.turns_open:
            reasons.append(f"{self.turns_open} caller turn(s) being decided")
        if self.speak_calls > len(self.tts.utterances):
            reasons.append("speech on its way to the text-to-speech")
        delivered = sum(e["type"] == "session.output_transcript.delta" for e in self.ws.from_server)
        if delivered < len(self.tts.spoken):
            reasons.append("output transcript events not yet delivered")
        return reasons + self.engine_pending()

    def engine_pending(self):
        """The adapter's, the core's and the gateway's own work in flight: the only reads of their internals.

        Re-point them when a change moves those objects. The presenter words, holds and speaks each result in its own
        task after the core has handed it over, so this method also reports the results the presenter has not spoken
        yet, under a reason of their own (:data:`RESULTS_PENDING`). The IM's task waits for the client's outputs itself
        (``TaskRun._wait_for_app``), and :meth:`instrument` notes each task in that wait.
        """
        engine, reasons = self.engine, []
        if engine.background:
            reasons.append("an append command still running in the engine")
        # The IM session is the gateway's one core's, which serves the session's conversation while it is open.
        im = next(iter(self.core_client.service.sessions.values()), None)
        # The work waits for the client only while a task is in its wait and its outputs have not come.
        waits_for_client = any(task.outputs_in is not None and not task.outputs_in.done() for task in self.tool_waits)
        if im is not None and im.busy and not waits_for_client:
            reasons.append(WORK_RUNNING)
        if engine.cascade is not None and engine.cascade.presenter.busy:
            reasons.append(RESULTS_PENDING)
        for session in self.gateway.sessions.values():
            if any(not subscriber.queue.empty() for subscriber in session.fanout.subscribers.values()):
                reasons.append("server events not yet sent to the client")
        return reasons

    # ------------------------------------------------------------------------------------- the golden
    def channels(self):
        """The call as recorded so far, by channel, before normalizing."""
        session, delegations, heard, said = [], {}, [], []
        for event in self.ws.from_server:
            kind = event["type"]
            if kind == "session.input_transcript.delta":
                heard.append(event)
            elif kind == "session.output_transcript.delta":
                said.append(event)
            elif kind == "session.delegation.created":
                delegations.setdefault(event["delegation"]["id"], []).append(event)
            elif kind == "response.event":
                delegations.setdefault(event["delegation_id"], []).append(event)
            elif kind != AUDIO_EVENT:
                session.append(event)
        return {
            "session": session,
            "delegations": delegations,
            "input_transcript": heard,
            "output_transcript": said,
            "spoken": list(self.tts.utterances),
            "frontend_requests": list(self.frontend.requests),
            "backend_requests": list(self.backend.requests),
        }

    def assert_golden(self, name):
        """Compare the call with ``recordings/<name>.json`` channel by channel, or record it.

        Only a run with ``LIVE_GOLDEN_RECORD=1`` writes the file, and it still fails when an existing file changed.
        """
        self.check_scripts_used()
        self.check_event_ids()
        recorded = normalize_channels(self.channels())
        path = GOLDEN_DIR / f"{name}.json"
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if os.environ.get(RECORD_VARIABLE) == "1":
            previous, golden = golden, {"name": name, "scenario": self.scenario(), "channels": recorded}
            GOLDEN_DIR.mkdir(exist_ok=True)
            path.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            diffs = changed_channels(name, previous.get("channels", {}), recorded) if previous else []
            if diffs:
                self.fail(
                    f"Recorded {path.name} again: {len(diffs)} channel(s) changed, and the file now holds this run. "
                    f"Check the diff, then run without {RECORD_VARIABLE}.\n\n" + "\n\n".join(diffs)
                )
        elif golden is None:
            self.fail(f"{path} does not exist; record it with {RECORD_VARIABLE}=1 (see the note in harness.py)")
        else:
            self.assertEqual(sorted(golden["channels"]), sorted(CHANNELS), f"{name}.json has the wrong channels")
            diffs = changed_channels(name, golden["channels"], recorded)
            if diffs:
                self.fail(
                    f"Golden {name} changed in {len(diffs)} channel(s). If the change is intended, record again "
                    f"with {RECORD_VARIABLE}=1.\n\n" + "\n\n".join(diffs)
                )

    def scenario(self):
        doc = inspect.getdoc(getattr(self, self._testMethodName)) or ""
        return " ".join(doc.split())

    def check_event_ids(self):
        """The wire: every server event has its own fresh ``event_id``; audio packets carry none.

        The normalizer writes every ``event_id`` as ``<event>``, so this check is what still notices two events that
        share one id, or an event that lost its id.
        """
        seen, problems = set(), []
        for number, event in enumerate(self.ws.from_server):
            kind, event_id = event.get("type"), event.get("event_id")
            if kind == AUDIO_EVENT:
                if "event_id" in event:
                    problems.append(f"event {number} ({kind}) carries an event_id: {event_id!r}")
            elif not isinstance(event_id, str) or not FRESH_EVENT_ID.fullmatch(event_id):
                problems.append(f"event {number} ({kind}) has no fresh event_id: {event_id!r}")
            elif event_id in seen:
                problems.append(f"event {number} ({kind}) repeats the event_id {event_id}")
            else:
                seen.add(event_id)
        if problems:
            self.fail("The server's event ids changed:\n" + "\n".join(problems))

    def check_scripts_not_overrun(self):
        for client in (self.frontend, self.backend):
            if client.overrun:
                last = (client.requests[client.overrun[0] - 1].get("messages") or [{}])[-1]
                self.fail(
                    f"The {client.role} client ran out of script at request {client.overrun[0]}, so the call made a "
                    f"model request the script did not expect. Its last message: {json.dumps(last)[:500]}"
                )

    def check_scripts_used(self):
        self.check_scripts_not_overrun()
        unused = {client.role: len(client.script) for client in (self.frontend, self.backend) if client.script}
        if unused:
            self.fail(f"Script steps left unused {unused}: the call did not go as scripted")
