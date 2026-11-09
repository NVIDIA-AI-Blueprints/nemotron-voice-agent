# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's voice: which run of the talker each one is, what its decision does, and what the backend sends."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from loguru import logger
from pipecat.frames.frames import InterruptionFrame, LLMMessagesAppendFrame
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, standard_message_text

from examples.frontend_backend_live.delegations import DelegationReader, ReadyResult
from examples.frontend_backend_live.display import NullDisplay
from examples.frontend_backend_live.handoffs import Acknowledgements, HandOffs
from examples.frontend_backend_live.heard_log import HeardLog
from examples.frontend_backend_live.models.decision import ACTIONS, CallSpec, TalkerDecision, decision_from_output_text
from examples.frontend_backend_live.spoken import Line, LineKind, SpokenLines
from examples.frontend_backend_live.talker import (
    CALL_FIELDS,
    Layout,
    TalkerCall,
    after_failure,
    floored,
    progress_line,
    relay_line,
)
from examples.frontend_backend_live.worknote import OpenWork, WorkNote
from examples.shared.ids import approx_tokens
from examples.shared.observe import log_observation

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator, LLMUserAggregator

    from examples.frontend_backend_live.config_manager.schema import LiveConfig
    from examples.frontend_backend_live.display import DisplaySink
    from examples.frontend_backend_live.prompts import FixedLines, PromptSet
    from examples.frontend_backend_live.talker import ResponseFormat, TalkerChat
    from gateways.live.ports import DelegationPort


def message_tokens(message: dict) -> int:
    """Return a cheap token estimate for one message, its role included: :func:`approx_tokens` of its JSON."""
    return approx_tokens(json.dumps(message, ensure_ascii=False))


@dataclass(frozen=True)
class Kind:
    """What a message of the talker's context is to the talker, and how its calls read it."""

    name: str
    role: str
    reference: bool = False
    turn: bool = False


CALLER = Kind("caller", "user", turn=True)
SPOKEN = Kind("spoken", "assistant")
CALL = Kind("call", "assistant")
CALL_RESULT = Kind("call_result", "tool")
RESULT = Kind("result", "user", reference=True)
APP_CONTEXT = Kind("app_context", "user", reference=True)
TYPED = Kind("typed", "user", reference=True)
# What the aggregators write, by role; any other message the voice did not write is a note no call reads.
WRITTEN = {"user": CALLER, "assistant": SPOKEN, "tool": CALL_RESULT}


@dataclass(frozen=True)
class Instruction(ReadyResult):
    """Instructions worded into a line, such as the greeting's; when they cannot be, ``fallback`` is said instead."""

    fallback: str

    @property
    def unworded(self) -> str:
        """The fallback line."""
        return self.fallback


@dataclass(eq=False)
class Owed:
    """Something to say: a worded item (its developer message goes into the context with its line) or a fixed line."""

    said: ReadyResult
    kind: LineKind
    message: dict | None = None
    answers: bool = False

    @property
    def worded(self) -> bool:
        """Whether it is worded by the relay, rather than said as written."""
        return self.message is not None


@dataclass(frozen=True)
class Turnover:
    """The end of a delegation, or its handover to ``taker``, applied in its turn."""

    delegation_id: str
    taker: str | None = None


LINE_KIND_ORDER = (LineKind.RESULT, LineKind.COMMENTARY, LineKind.NOTICE)


def as_written(items: Sequence[Owed]) -> str:
    """Return the items as the backend or the application wrote them, in order, each starting a line.

    The text-to-speech strips a list marker or a heading only at a line's start, so each item keeps its own line.
    """
    return "\n".join(text for text in (item.said.unworded.strip() for item in items) if text)


@dataclass
class Routing:
    """A routing call: a decision on the caller's undecided turns, an instruction, or a caller turn with no words."""

    covers: tuple[dict, ...]
    request: str
    work_open: bool

    streams: ClassVar[bool] = True

    def cut_by_result(self) -> bool:
        """Return whether a result cuts it: only a decision on caller turns reads the work note a result changes."""
        return bool(self.covers)

    def moved(self, voice: Voice) -> bool:
        """Return ``False``: only a result cuts a decision."""
        return False

    def holds(self, delegation_id: str) -> bool:
        """Return ``False``: it words no item."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Carry the decision out; a hand-off reserves its work-note entry with nothing awaited before it."""
        # Words already said answer the turn, but a ``delegate`` call made with them still hands it over.
        if call.said and not (decision.action == "delegate" and decision.call_id):
            decision = TalkerDecision("speak", call.said, decision.diagnostics)
        decision = floored(decision, work_open=self.work_open, holding=voice.lines.holding)
        voice.decided_by(self)
        voice.observe("talker.completed", action=decision.action, **decision.diagnostics)
        return voice.actions[decision.action](self, call, decision)

    def dropped(self, voice: Voice) -> None:
        """Leave its turns for the run the result brings, which decides them with the result in view."""
        voice.observe("talker.cut_off")

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Leave its turns undecided, unless its words went to speech."""
        if call.said:
            voice.decided_by(self)

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Take its turns as decided; return whether the fallback is due: not when its words went to speech."""
        voice.decided_by(self)
        return not call.said

    def recover(self, voice: Voice) -> None:
        """Say the holding line while work is open; otherwise delegate the turn with no lead-in."""
        decision = after_failure({"recovery": "talker_failed"}, work_open=self.work_open, holding=voice.lines.holding)
        voice.actions[decision.action](self, None, decision)


@dataclass
class Relay:
    """A relay: what is owed, worded into one line with the writer's rules."""

    batch: tuple[Owed, ...]
    content: str
    history: list[dict]
    note: str = ""
    version: int = 0

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Return ``True``: its line is held to its end, so a result that arrives first is worded with these items."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Return whether what it words moved since it began: a new entry owed, its note, or the voice's version."""
        return bool(voice.owed) or voice.version != self.version or voice._update(self.batch)[1] != self.note

    def holds(self, delegation_id: str) -> bool:
        """Return whether it words an item of ``delegation_id``, whose end or handover then waits for its re-word."""
        return any(item.worded and item.said.delegation_id == delegation_id for item in self.batch)

    def words(self, voice: Voice, decision: TalkerDecision) -> str:
        """Return the worded line, or the worded items as written when it does not pass the checks."""
        line = relay_line(decision, self.content, self.history)
        if line is None:
            voice.observe("commentary.answer_forced", talker_action=decision.action, results=len(self.batch))
            line = as_written([item for item in self.batch if item.worded])
        return line

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Return the worded line, or the items as written when it fails the checks; nothing once it moved."""
        if self.moved(voice):
            self.dropped(voice)
            return ""
        delivered = voice.delivered(self.batch, voice.joined(self.batch, self.words(voice, decision)))
        return voice.hand(delivered) if delivered else ""

    def dropped(self, voice: Voice) -> None:
        """Owe its items again, first: the run the result brings words them with it, in one line."""
        voice.owe_again(self.batch)

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Owe its items again."""
        voice.owe_again(self.batch)

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Return ``True``: the items are said as written."""
        return True

    def recover(self, voice: Voice) -> None:
        """Say the items as written, with no model call."""
        line = voice.delivered(self.batch, voice.joined(self.batch, as_written([i for i in self.batch if i.worded])))
        if line is not None:
            voice.say(line)


@dataclass
class Progress:
    """A progress line about one delegation's quiet work."""

    entry: OpenWork
    version: int
    history: list[dict]

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Return ``True``: a result moves what the line is about."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Return whether a new item or the voice's version moved since it was offered."""
        return voice.version != self.version or any(isinstance(entry, Owed) for entry in voice.owed)

    def holds(self, delegation_id: str) -> bool:
        """Return ``False``: it words no item."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Say the line only if nothing moved while it was worded; its news is taken either way then."""
        delegation_id = self.entry.id
        if self.moved(voice) or not voice.worknote.quiet(delegation_id):
            voice.observe("progress.dropped", delegation_id=delegation_id)
            return ""
        voice.worknote.worded(delegation_id)
        text = progress_line(decision, self.entry.progress, self.history)
        if not text:
            voice.observe("progress.silent", delegation_id=delegation_id)
            return ""
        voice.observe("progress.said", delegation_id=delegation_id)
        return voice.hand(Line(LineKind.STATUS, text, (delegation_id,), tells=(delegation_id,), again=False))

    def dropped(self, voice: Voice) -> None:
        """Keep its news for the next quiet spell."""
        voice.observe("progress.dropped", delegation_id=self.entry.id)

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Do nothing: a progress line holds no fact to keep."""

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Return ``False``: nothing is said."""
        return False

    def recover(self, voice: Voice) -> None:
        """Do nothing."""


@dataclass
class Saying:
    """A line said with no model call: a lead-in, a fixed line, a line said once more."""

    line: Line

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Return ``False``: it asks no model."""
        return False

    def moved(self, voice: Voice) -> bool:
        """Return ``False``: it asks no model."""
        return False

    def holds(self, delegation_id: str) -> bool:
        """Return ``False``: what it says is delivered already."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Hand the line to speech."""
        return voice.hand(self.line)

    def dropped(self, voice: Voice) -> None:
        """Do nothing: it is never cut."""

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Do nothing: the cut that cancelled it settles it as not heard."""

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Return ``False``: it sends no request."""
        return False

    def recover(self, voice: Voice) -> None:
        """Do nothing."""


class Floor:
    """When the caller speaks, and how long the voice has been idle."""

    def __init__(self, clock: Callable[[], float]):
        """Start idle, with the caller silent."""
        self.clock = clock
        self.caller_speaking = False
        self.caller_starts = 0
        self.idle = True
        self.idle_since = clock()
        self._changed = asyncio.Event()

    def caller(self, speaking: bool) -> None:
        """Note that the caller started or stopped speaking."""
        self.caller_speaking = speaking
        self.caller_starts += speaking
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def settle(self, idle: bool) -> None:
        """Note whether the voice is idle now: the caller silent, no call running, nothing in flight or due."""
        if idle and not self.idle:
            self.idle_since = self.clock()
        self.idle = idle

    async def wait_caller_silent(self) -> None:
        """Return once the caller is silent: at once when they are."""
        while self.caller_speaking:
            await self._changed.wait()


@dataclass(eq=False)
class EarlyLine:
    """What is owed, worded early: the items, the stamp of what the wording read, and the line once it is worded."""

    batch: tuple[Owed, ...]
    stamp: tuple
    task: asyncio.Task | None = None
    words: str | None = None

    def cut(self) -> None:
        """Stop the wording if it is still under way."""
        if self.task is not None:
            self.task.cancel()


def _logged_crash(task: asyncio.Task) -> None:
    """Log a background task that stopped on an error, which its close only gathers."""
    if not task.cancelled() and task.exception() is not None:
        logger.opt(exception=task.exception()).error(f"{task.get_name()} stopped on an error")


class Voice:
    """One call's voice: chooses each run of the talker and acts on its decision; says what the backend sends."""

    def __init__(
        self,
        *,
        config: LiveConfig,
        prompts: PromptSet,
        lines: FixedLines,
        json_rules: str,
        empty_turn_note: str,
        delegation: DelegationPort,
        item_text: Callable[[dict], str],
        reports_sentences: bool = True,
        display: DisplaySink | None = None,
        observe: Callable[..., None] = log_observation,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Wire the call's records and its delegation part; the talker is bound by :meth:`bind`."""
        self.config, self.prompts, self.lines, self.delegation = config, prompts, lines, delegation
        self.empty_turn_note, self.observe, self.clock = empty_turn_note, observe, clock
        self.layout = Layout(prompts, json_rules)
        self.display = display or NullDisplay()
        self.routing_spec = CallSpec("talker", ACTIONS, prompts.call_hints.talker)
        self.answer_spec = CallSpec("writer", ("speak",), prompts.call_hints.answer)
        self.progress_spec = CallSpec("writer", ("speak",), prompts.call_hints.progress)
        self.actions: dict[str, Callable[[Routing, TalkerCall | None, TalkerDecision], str]] = {
            "speak": self._spoke,
            "delegate": self._delegated,
            "wait": self._waited,
        }
        self.floor = Floor(clock)
        self.worknote = WorkNote(prompts, tracks=delegation.reports_events, clock=clock, observe=observe)
        self.heard_log = HeardLog(prompts)
        self.spoken = SpokenLines(self, reports_sentences=reports_sentences, observe=observe)
        self.acknowledgements = Acknowledgements(self.floor, self._lead_in, observe=observe)
        self.handoffs = HandOffs(
            delegation,
            self.heard_log,
            self.worknote,
            self.display,
            self.acknowledgements,
            refused=lambda: self.notice(None, lines.refusal),
            observe=observe,
        )
        self.reader = DelegationReader(
            self.worknote,
            self,
            self.display,
            lines,
            display_label=config.display_label,
            app_wait_seconds=config.delegation.app_wait_seconds,
            client_tool_timeout_seconds=config.client_tool_timeout_seconds,
            item_text=item_text,
            observe=observe,
            outcome_arrived=self.acknowledgements.drop,
        )
        self.worknote.on_answer = self.result_arrived
        self.talker: TalkerChat | None = None
        self.routing_format: ResponseFormat | None = None
        self.context: LLMContext | None = None
        self.instructions: list[str] = []
        self.seed_roles: list[str] = []
        self.kinds: dict[int, tuple[dict, Kind]] = {}
        self.decided: dict[int, dict] = {}
        self.noted: dict[int, dict] = {}
        self.unsaid: set[int] = set()
        self.sent: list[dict] = []
        self._writes: list[dict] = []
        self.owed: deque[Owed | Turnover] = deque()
        self.early: EarlyLine | None = None
        self.waiting_lines: deque[Line] = deque()
        self.lead_ins: dict[str, Line] = {}
        self.instruction_due = False
        self.progress_due: OpenWork | None = None
        self.call: TalkerCall | None = None
        self.failures: deque[TalkerCall] = deque()
        self.responses = 0
        self.run_requested = False
        self.version = 0
        self._offered_at = float("-inf")
        self._made = 0
        self._tasks: set[asyncio.Task] = set()
        self._quiet_task: asyncio.Task | None = None
        self._closed = False
        delegation.listen(self.reader)

    def seed(self, messages: Sequence[dict]) -> list[dict]:
        """Take the host's seed: its system and developer texts are the instructions; the rest starts the context.

        Instructions stay out of the context, so the bound never drops them; a labeled user message is reference data.
        """
        conversation = []
        labels = {self.prompts.result_label: RESULT, self.prompts.context_label: APP_CONTEXT}
        for message in messages:
            text = standard_message_text(message)
            if message.get("role") in ("system", "developer"):
                self.instructions.append(text)
                self.seed_roles.append(message["role"])
                continue
            kind = next((kind for label, kind in labels.items() if text.startswith(label)), None)
            if message.get("role") == "user" and kind is not None:
                message = {**message, "role": "developer"}
                self._register(message, kind)
            conversation.append(message)
            self.decided[id(message)] = message
        return conversation

    def show(self, display: DisplaySink) -> None:
        """Show each delegation on ``display``, a sink made once the pipeline's worker is built."""
        self.display = self.reader.display = self.handoffs.display = display

    def bind(self, talker: TalkerChat, context: LLMContext) -> None:
        """Run ``talker`` on ``context``; a routing call asks in the talker's routing format, framed in its texts."""
        self.talker, self.context = talker, context
        self.routing_format = talker.routing_format(self.prompts)
        self.layout.routing_texts = self.routing_format.routing_texts(self.prompts)
        talker.bind(self)

    def start(self) -> None:
        """Start the quiet task that offers progress lines, once; the host that built the voice starts it."""
        progress_after = self.config.delegation.progress_after_seconds
        if self._quiet_task is None and progress_after is not None:
            self._quiet_task = asyncio.create_task(self._quiet(progress_after), name="delegation-progress")
            self._quiet_task.add_done_callback(_logged_crash)

    async def close(self) -> None:
        """Ignore later delegation events, drop what waits to be said, and stop every task. Safe twice."""
        self._closed = True
        self.reader.close()
        self.acknowledgements.drop_all()
        self.handoffs.close()
        if self.early is not None:
            self.early.cut()
        tasks = [*self._tasks, *([self._quiet_task] if self._quiet_task else [])]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.owed.clear()
        self.waiting_lines.clear()

    def _register(self, message: dict, kind: Kind) -> None:
        self.kinds[id(message)] = (message, kind)

    def kind_of(self, message: dict) -> Kind | None:
        """Return what a context message is: one the voice wrote, one an aggregator wrote, or ``None`` for a note."""
        registered = self.kinds.get(id(message))
        if registered is not None and registered[0] is message:
            return registered[1]
        if message.get("tool_calls"):
            return CALL
        return WRITTEN.get(message.get("role"))

    @staticmethod
    def as_read(message: dict, kind: Kind) -> dict:
        """Return ``message`` as the talker's calls read it: its kind's role, its text, and a tool call's fields."""
        fields = {name: message[name] for name in CALL_FIELDS if name in message}
        return {"role": kind.role, "content": standard_message_text(message), **fields}

    def conversation(self, *, labeled: bool = False) -> list[dict]:
        """Return a view of what the talker's calls read, each message with its kind; it changes nothing.

        ``labeled``: the routing call's view, with the caller label on a caller turn right after reference data.
        """
        history, after_reference = [], False
        for message in self._read():
            kind = self.kind_of(message)
            read = self.as_read(message, kind)
            if labeled and after_reference and kind.turn:
                read["content"] = self.prompts.caller_label + read["content"]
            after_reference = kind.reference
            history.append({**read, "kind": kind.name})
        return history

    def _read(self) -> list[dict]:
        """Return what the calls read: the context and the voice's appends on their way, but notes and unsaid items."""
        messages = list(self.context.get_messages()) if self.context else []
        held = {id(message) for message in messages}
        on_the_way = [message for message in self.sent if id(message) not in held]
        return [m for m in [*messages, *on_the_way] if self.kind_of(m) is not None and id(m) not in self.unsaid]

    def bound(self) -> None:
        """Drop the oldest turn groups once the context passes ``memory.compact_at`` of ``memory.context_tokens``.

        The count holds the instructions and each line in flight whole; each caller turn opens a group, and the newest
        group never leaves. No model is called.
        """
        if self.context is None:
            return
        memory = self.config.memory
        limit = memory.compact_at * memory.context_tokens
        messages = self.context.get_messages()
        held = {id(message) for message in messages}
        self.sent = [message for message in self.sent if id(message) not in held]
        pinned = [
            {"role": self.seed_roles[i] if i < len(self.seed_roles) else "developer", "content": text}
            for i, text in enumerate(self.instructions)
        ]
        tokens = sum(map(message_tokens, pinned)) + sum(map(self._tokens, self.sent))
        tokens += sum(
            message_tokens({"role": SPOKEN.role, "content": line.text}) for line in self.spoken.lines.values()
        )
        groups: list[list[int]] = []
        for index, message in enumerate(messages):
            if id(message) in self.unsaid:
                continue
            kind = self.kind_of(message)
            if (kind is not None and kind.turn) or not groups:
                groups.append([])
            groups[-1].append(index)
        weights = [sum(self._tokens(messages[index]) for index in group) for group in groups]
        tokens += sum(weights)
        dropped: set[int] = set()
        for group, weight in zip(groups[:-1], weights[:-1], strict=True):
            if tokens <= limit:
                break
            dropped.update(group)
            tokens -= weight
        if not dropped:
            return
        for index in dropped:
            key = id(messages[index])
            for records in (self.kinds, self.decided, self.noted):
                records.pop(key, None)
        self.context.set_messages([message for index, message in enumerate(messages) if index not in dropped])

    def _tidy(self) -> None:
        """Ready the context for a request: take out the items not said yet, and bound it."""
        self._take_out(self.unsaid)
        self.bound()

    def _tokens(self, message: dict) -> int:
        """Return the estimated tokens of ``message`` as the calls read it; none for a note or an item not said."""
        kind = self.kind_of(message)
        if kind is None or id(message) in self.unsaid:
            return 0
        return message_tokens(self.as_read(message, kind))

    def runtime_data(self) -> str:
        """Return the routing call's runtime data now, with no turn named."""
        return self.layout.runtime_data(self.worknote.line())

    def _undecided(self) -> list[dict]:
        """Return the caller turns no routing call decided yet, oldest first."""
        messages = self.context.get_messages() if self.context else ()
        return [m for m in messages if (kind := self.kind_of(m)) and kind.turn and id(m) not in self.decided]

    def _append(self, message: dict, kind: Kind, *, run: bool) -> None:
        """Queue ``message`` for the talker's context; ``run``: run the talker then, unless one is already asked for."""
        self._register(message, kind)
        self.sent.append(message)
        self._queue(LLMMessagesAppendFrame(messages=[message], run_llm=run and not self.run_requested))
        self.run_requested = self.run_requested or run

    def request_run(self) -> None:
        """Run the talker once nobody speaks: an empty append, which the assistant aggregator holds and bundles."""
        if not self.run_requested:
            self.run_requested = True
            self._queue(LLMMessagesAppendFrame(messages=[], run_llm=True))

    def _queue(self, frame) -> None:
        """Queue ``frame`` at the talker, uninterruptible, so it passes any interruption on the way."""
        if self.talker is None or self._closed:
            return
        frame.interruptible = False
        task = asyncio.get_running_loop().create_task(self.talker.queue_frame(frame))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def next_call(self, context: LLMContext) -> TalkerCall | None:
        """Return the call this run of the talker makes; ``None``: no request.

        A caller turn, an instruction or an empty turn's note makes a routing call; then come a waiting line, what is
        owed in one relay, and a progress line.
        """
        self.run_requested = False
        call = None
        self._settle_results(context)
        note = self._empty_turn(context)
        undecided = self._undecided()
        if note or undecided or self.instruction_due:
            call = await self._routing(undecided, note)
        elif self.waiting_lines:
            call = self.saying(self._next_line())
        elif any(isinstance(entry, Owed) for entry in self.owed):
            try:
                await self._worded_early()
            except asyncio.CancelledError:
                self._after_call()
                raise
            if batch := self._take():
                call = self._said_by(batch)
        elif self.progress_due is not None:
            call = self._progress()
        self.call = call
        if call is not None and call.spec is not None:
            call.deadline = asyncio.get_running_loop().time() + self.config.reliability.talker_attempt_timeout_seconds
        self._moved()
        return call

    def _settle_results(self, context: LLMContext) -> None:
        """Write each routing tool result Pipecat deferred into its call's placeholder, as if it came first."""
        messages = context.get_messages()
        placeholders = {m.get("tool_call_id"): m for m in messages if self.kind_of(m) is CALL_RESULT}
        for message in messages:
            payload = async_tool_messages.parse_message(message) if self.kind_of(message) is None else None
            if payload is not None and payload.kind == "final" and payload.tool_call_id in placeholders:
                placeholders[payload.tool_call_id]["content"] = payload.result

    def _empty_turn(self, context: LLMContext) -> str:
        """Take Pipecat's notes out of the context; return the empty-turn note when the run is that turn's own."""
        messages = list(context.get_messages())
        notes = [message for message in messages if self.kind_of(message) is None]
        if not notes:
            return ""
        own = notes[-1] is messages[-1] and messages[-1].get("content") == self.empty_turn_note
        context.set_messages([message for message in messages if self.kind_of(message) is not None])
        return self.empty_turn_note if own else ""

    async def _routing(self, undecided: list[dict], note: str = "") -> TalkerCall:
        """Return the routing call on the caller's undecided turns, an instruction or an empty turn's ``note``."""
        for message in undecided:
            if id(message) not in self.noted:
                self.heard_log.caller(standard_message_text(message))
                self.noted[id(message)] = message
        turn = " ".join(standard_message_text(message) for message in undecided)
        request = turn or (self.instructions[-1] if self.instruction_due else "")
        plan = Routing(tuple(undecided), request, bool(self.worknote.open()))
        self._tidy()
        routing = self.layout.routing("\n\n".join(self.instructions), self.worknote.line(), turn, note)
        return TalkerCall(
            spec=self.routing_spec, request=routing, streams=plan.streams, format=self.routing_format, plan=plan
        )

    def _take(self) -> list[Owed]:
        """Take everything owed, oldest first; apply each end or handover whose item the batch does not word."""
        batch: list[Owed] = []
        while self.owed:
            entry = self.owed.popleft()
            if isinstance(entry, Owed):
                batch.append(entry)
            elif not any(item.worded and item.said.delegation_id == entry.delegation_id for item in batch):
                self._turnover(entry)
        return batch

    def _turnover(self, entry: Turnover) -> None:
        if entry.taker is None:
            self.spoken.forget(entry.delegation_id)
        else:
            self.spoken.carry_told(entry.delegation_id, entry.taker)

    def _said_by(self, batch: list[Owed]) -> TalkerCall | None:
        """Return the call that says ``batch``: the relay's, or the fixed lines as written when none is worded."""
        if any(item.worded for item in batch):
            return self.relay(batch)
        line = self.delivered(batch, self.joined(batch, ""))
        return self.saying(line) if line else None

    def relay(self, batch: list[Owed]) -> TalkerCall | None:
        """Return the call that says ``batch``: its early wording while its stamp holds, else a relay."""
        early, self.early = self.early, None
        if early is not None:
            early.cut()
            if early.words is not None and early.stamp == self._stamp(batch):
                self.observe("answer.worded_early", items=len(batch))
                line = self.delivered(batch, self.joined(batch, early.words))
                return self.saying(line) if line else None
        self._tidy()
        return self._relay_call(batch, [dict(message) for message in self.context.get_messages()])

    def _relay_call(self, batch: Sequence[Owed], history: list[dict]) -> TalkerCall:
        """Return the relay that words ``batch``: the writer's frame and rules, ``history``, then the update."""
        content, note = self._update(batch)
        relay = self.layout.relay("\n\n".join(self.instructions), content, note)
        plan = Relay(tuple(batch), content, history, note, self.version)
        return TalkerCall(spec=self.answer_spec, request=relay, format=self.talker.response_format, plan=plan)

    def _update(self, batch: Sequence[Owed]) -> tuple[str, str]:
        """Return the update a relay of ``batch`` words: its items' text, and its note on what the caller heard."""
        worded = [item for item in batch if item.worded]
        delegations = list(dict.fromkeys(item.said.delegation_id for item in worded if item.said.delegation_id))
        told = " ".join(text for each in delegations if (text := self.spoken.told_text(each)))
        note = self.layout.note(told or None, self.worknote.newer_requests(delegations))
        return "\n\n".join(item.said.speak for item in worded), note

    def _stamp(self, batch: Sequence[Owed]) -> tuple:
        """Return what a wording of ``batch`` stands on: its items, its update's note, and the voice's version."""
        return tuple(batch), self._update(batch)[1], self.version

    def _deciding(self) -> bool:
        """Return whether a caller turn or an instruction waits for its routing call, or is being decided."""
        return self.instruction_due or bool(self._undecided())

    def _word_early(self) -> None:
        """Word what is owed now, out of band, unless a wording of it stands.

        A wording starts at once, even while a caller turn is decided, but none while a waiting line is due. A moved
        stamp cuts the wording under way, and its new one waits until no caller turn or instruction waits or is
        decided. Each reads the lines in flight as said, as they will be once they play.
        """
        if self.talker is None or self._closed:
            return
        batch = tuple(entry for entry in self.owed if isinstance(entry, Owed))
        early = self.early
        if not any(item.worded for item in batch):
            return
        stamp = self._stamp(batch)
        if early is not None:
            if early.stamp == stamp:
                return
            early.cut()
            if self._deciding():
                return
        if self.waiting_lines:
            return
        in_flight = [{"role": SPOKEN.role, "content": line.text} for line in self.spoken.lines.values()]
        self._tidy()
        history = [*(dict(message) for message in self._read()), *in_flight]
        call = self._relay_call(batch, history)
        self.early = EarlyLine(batch, stamp)
        words = self._words(self.early, call, LLMContext(list(history)))
        self.early.task = asyncio.get_running_loop().create_task(words, name="early-wording")

    async def _words(self, early: EarlyLine, call: TalkerCall, context: LLMContext) -> None:
        """Word ``early`` with ``call``, run out of band on ``context``, or as written when the call fails."""
        try:
            async with asyncio.timeout(self.config.reliability.talker_attempt_timeout_seconds):
                reply = await self.talker.run_call(call, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Wording early failed ({type(exc).__name__}: {exc}); holding the items as written")
            self.observe("commentary.recovered", error_type=type(exc).__name__, results=len(early.batch))
            early.words = as_written([item for item in early.batch if item.worded])
            return
        early.words = call.plan.words(self, decision_from_output_text(reply or "", {}))

    async def _worded_early(self) -> None:
        """Wait for the wording under way of what is owed now, so the relay says its line rather than word it again."""
        while (early := self.early) is not None and early.task is not None and not early.task.done():
            if early.stamp != self._stamp(tuple(entry for entry in self.owed if isinstance(entry, Owed))):
                return
            await asyncio.wait({early.task})

    def saying(self, line: Line) -> TalkerCall:
        """Return a call that says ``line`` with no model call."""
        return TalkerCall(text=line.text, plan=Saying(line))

    @staticmethod
    def joined(batch: Sequence[Owed], words: str) -> str:
        """Return the batch's line: the worded items' words at the first one's place, the fixed lines as written."""
        parts: list[str] = []
        placed = False
        for item in batch:
            if not item.worded:
                parts.append(item.said.speak)
            elif not placed:
                parts.append(words)
                placed = True
        return " ".join(part.strip() for part in parts if part.strip())

    def delivered(self, batch: Sequence[Owed], text: str) -> Line | None:
        """Return the line that says ``batch``, its items said and what it told worded in; ``None`` with no words."""
        kinds = {item.kind for item in batch}
        kind = next(each for each in LINE_KIND_ORDER if each in kinds)
        ids = tuple(dict.fromkeys(item.said.delegation_id for item in batch if item.said.delegation_id))
        for item in batch:
            if item.worded:
                if item.said.delegation_id:
                    self.spoken.forget(item.said.delegation_id)
                self.unsaid.discard(id(item.message))
        if not text:
            self._done(batch)
            return None
        self.observe("answer.said", items=len(batch), delegations=list(ids))
        return Line(kind, text, ids, items=tuple(batch), results=any(item.answers for item in batch))

    def _progress(self) -> TalkerCall | None:
        """Return a progress line about the quiet work offered, worded from the work note, if it is still quiet."""
        entry, self.progress_due = self.progress_due, None
        if not self.worknote.quiet(entry.id):
            return None
        update = self.worknote.offer(entry.id)
        self.observe("progress.offered", delegation_id=entry.id)
        note = self.layout.note(self.spoken.told_text(entry.id))
        self._tidy()
        progress = self.layout.progress("\n\n".join(self.instructions), update, note)
        plan = Progress(entry, self.version, [dict(message) for message in self.context.get_messages()])
        return TalkerCall(spec=self.progress_spec, request=progress, format=self.talker.response_format, plan=plan)

    def speaking(self, call: TalkerCall, text: str) -> None:
        """Add a routing ``speak``'s words to this response's line as they go to speech."""
        line = self.spoken.lines.get(self.responses)
        if line is None:
            self.hand(Line(LineKind.REPLY, text))
        else:
            line.text += text
            self.heard_log.replace_agent(line.number, line.text)

    def finished(self, call: TalkerCall, decision: TalkerDecision | None, held: str) -> str:
        """Act on the call's decision (its plan's); return the words the response lets out at its end."""
        self.call = None
        try:
            return call.plan.finished(self, call, decision)
        finally:
            self._after_call()

    def _spoke(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """Say a reply: streamed already, or let out now (the holding line, speech decoded before the action)."""
        line = self.spoken.lines.get(self.responses) if call is not None and call.said else None
        if line is not None:
            if not decision.diagnostics.get("recovery"):
                line.text = decision.speech
                self.heard_log.replace_agent(line.number, line.text)
            self.bound()
            self._move()
            return ""
        reply = Line(LineKind.REPLY, decision.speech)
        if call is None:
            self.say(reply)
            return ""
        return self.hand(reply)

    def _delegated(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """Hand the turn to the delegation part with the decision's speech as its lead-in; with no request, nothing.

        Words of the call already said answer the turn as a reply, and the hand-off says no lead-in of its own.
        """
        lead_in, text = decision.speech, ""
        if call is not None and call.said:
            lead_in, text = "", self._spoke(plan, call, TalkerDecision("speak", call.said, decision.diagnostics))
        if plan.request:
            decided_at = self.floor.caller_starts
            self.handoffs.delegate(plan.request, lead_in, decided_at=decided_at, call_id=decision.call_id)
        return text

    def _waited(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """Say nothing for the turn."""
        self.observe("talker.waited")
        return ""

    def decided_by(self, plan: Routing) -> None:
        """Mark the plan's caller turns as decided, and any instruction read with them."""
        for message in plan.covers:
            self.decided[id(message)] = message
        self.instruction_due = False

    def hand(self, line: Line) -> str:
        """Hand ``line`` to speech as the response in progress, and return its words.

        The heard log notes it at once, so a delegation made while it plays carries it; playout corrects the entry.
        """
        said_as, line.number = line.number, self.responses
        if line.retried:
            self.heard_log.said_again(said_as, line.number)
        else:
            self.heard_log.agent(line.number, line.text, result=line.results)
        self.spoken.handed(line)
        self._write_with(line)
        self.bound()
        self._move()
        return line.text

    def _write_with(self, line: Line) -> None:
        """Put the items ``line`` says on their way into the context, out of where a line nobody heard left them."""
        self._writes = [item.message for item in line.items if item.worded]
        if self._writes:
            self._take_out({id(message) for message in self._writes})
            self.sent += self._writes

    def _take_out(self, ids: set[int]) -> None:
        """Remove the messages ``ids`` from the talker's context and from the voice's appends on their way."""
        if not ids:
            return
        if self.context is not None and any(id(message) in ids for message in self.context.get_messages()):
            self.context.set_messages([message for message in self.context.get_messages() if id(message) not in ids])
        self.sent = [message for message in self.sent if id(message) not in ids]

    def written(self) -> list[dict]:
        """Return the developer messages that go into the context right before the line just handed."""
        writes, self._writes = self._writes, []
        return writes

    def dropped(self, call: TalkerCall) -> None:
        """End the call, which a result reached before it said anything."""
        self.call = None
        call.plan.dropped(self)
        self._after_call()

    def cancelled(self, call: TalkerCall) -> None:
        """End the call an interruption cancelled."""
        if self.call is call:
            self.call = None
        call.plan.cancelled(self, call)
        self._after_call()

    def failed(self, call: TalkerCall, error: BaseException) -> None:
        """Note the failed call; its fallback runs from the pipeline's error event (:meth:`recover`)."""
        if self.call is call:
            self.call = None
        logger.warning(f"Talker failed ({type(error).__name__}: {error}); recovering the call")
        self.observe("talker.recovered", error_type=type(error).__name__)
        if call.plan.failed(self, call):
            self.failures.append(call)
        self._after_call()

    def recover(self) -> None:
        """Run each failed call's fallback once Pipecat reported the failure."""
        while self.failures:
            self.failures.popleft().plan.recover(self)
        self._after_call()

    async def hand_off(self, call_id: str) -> bool | None:
        """Hand over the turn the ``delegate`` call ``call_id`` decided; return whether the part accepted it."""
        return await self.handoffs.hand_off(call_id)

    def response_started(self) -> int:
        """Return the number of the response the talker starts."""
        self.responses += 1
        return self.responses

    def response_ended(self) -> int:
        """Return the number of the response the talker ends, handing over what no call's handler will run."""
        self.handoffs.after_response()
        return self.responses

    def interrupted(self) -> None:
        """Drop the responses started so far that have not played, and the lines waiting to be said again."""
        self.spoken.interrupted(self.responses)
        self._drop_retries()

    def _drop_retries(self) -> None:
        """Settle each line waiting to be said again as not heard: it goes with the interruption."""
        retries = [line for line in self.waiting_lines if line.retried]
        if retries:
            self.waiting_lines = deque(line for line in self.waiting_lines if not line.retried)
            for line in retries:
                line.heard = ""
                self.not_heard(line)

    def caller_started(self) -> None:
        """Note the caller speaking: drop the lead-ins not said yet, and say nothing meanwhile."""
        self.floor.caller(True)
        self.acknowledgements.drop_all()
        self._withdraw()
        self._moved()

    def caller_stopped(self) -> None:
        """Note that the caller stopped speaking."""
        self.floor.caller(False)
        self._moved()

    def result_arrived(self) -> None:
        """Cut the call in progress that has said nothing, when a result changes what it read."""
        call = self.call
        if call is not None and not call.said and call.plan.cut_by_result():
            call.cut = True

    def _after_call(self) -> None:
        """Ask for one more run once nobody speaks, when anything is still due."""
        self._moved()
        if self._due() and not self.run_requested:
            self.request_run()

    def _due(self) -> bool:
        """Return whether anything waits for a run of the talker."""
        owed = any(isinstance(entry, Owed) for entry in self.owed)
        return bool(owed or self.waiting_lines or self.progress_due or self.instruction_due or self._undecided())

    def _moved(self) -> None:
        call = self.call
        if call is not None and not call.said and call.plan.moved(self):
            call.cut = True
        idle = not self.floor.caller_speaking and self.call is None and not self.spoken.in_flight() and not self._due()
        self.floor.settle(idle)
        self._word_early()

    def _move(self) -> None:
        """Note that something every wording reads moved: a line said or taken back, an instruction, thinking, a cut."""
        self.version += 1
        self._moved()

    def heard(self, line: Line, cut: bool) -> None:
        """Settle ``line`` as heard, even in part: a cut line keeps what played, and its items are said."""
        if cut:
            self.heard_log.cut(line.number, line.heard)
        self._done(line.items)
        self._moved()

    def not_heard(self, line: Line) -> None:
        """Settle ``line`` as not heard: it leaves the heard log, and its items are owed again, first."""
        self.heard_log.unsay(line.number)
        if line.items:
            self.owe_again(line.items)
            self.observe("answer.requeued", line=line.number, items=len(line.items))
            self.request_run()
        self._move()

    def say_again(self, line: Line) -> None:
        """Say ``line`` once more, behind the lines already waiting: it produced no audio."""
        self.say(line)

    def owe_again(self, items: Sequence[Owed]) -> None:
        """Owe ``items`` again, first; until a line says them, their messages stay out of the context."""
        for item in reversed(items):
            self.owed.appendleft(item)
            if item.worded:
                self.unsaid.add(id(item.message))
        self._take_out({id(item.message) for item in items if item.worded})

    def _done(self, items: Sequence[Owed]) -> None:
        """Count each result among ``items`` as said in the work note."""
        for item in items:
            if item.answers and item.said.delegation_id:
                self.worknote.said(item.said.delegation_id)

    def say(self, line: Line) -> None:
        """Say ``line`` with no model call, in its turn once nobody speaks."""
        self.waiting_lines.append(line)
        self.request_run()

    def _next_line(self) -> Line:
        line = self.waiting_lines.popleft()
        for delegation_id in line.delegation_ids:
            if self.lead_ins.get(delegation_id) is line:
                del self.lead_ins[delegation_id]
        return line

    def _lead_in(self, delegation_id: str, text: str) -> None:
        """Say a delegation's lead-in, which tells the caller about it."""
        line = Line(LineKind.LEAD_IN, text, (delegation_id,), tells=(delegation_id,))
        self.lead_ins[delegation_id] = line
        self.say(line)

    def _withdraw(self) -> None:
        """Take back the lead-ins not said yet, which the caller's words would cut: a later decision answers them."""
        withdrawn, self.lead_ins = list(self.lead_ins.values()), {}
        self.waiting_lines = deque(line for line in self.waiting_lines if all(line is not w for w in withdrawn))
        for line in withdrawn:
            self.observe("acknowledgement.dropped", delegation_id=line.delegation_ids[0])

    def _own_id(self, kind: str) -> str:
        self._made += 1
        return f"{kind}_{self._made}"

    def _owe(self, entry: Owed | Turnover) -> None:
        """Owe ``entry``; apply an end or handover at the head, unless the call under way words its item."""
        if self._closed:
            return
        self.owed.append(entry)
        while self.owed and isinstance(head := self.owed[0], Turnover) and not self._holds(head.delegation_id):
            self._turnover(self.owed.popleft())
        self._moved()

    def _holds(self, delegation_id: str) -> bool:
        """Return whether the call under way words an item of ``delegation_id``."""
        return self.call is not None and self.call.plan.holds(delegation_id)

    def ready(self, result: ReadyResult) -> None:
        """Owe a ready result, a ``final_answer`` item, and run once nobody speaks; its line writes it in."""
        message = {"role": "developer", "content": self.prompts.result_label + result.speak}
        self._register(message, RESULT)
        self.unsaid.add(id(message))
        self._owe(Owed(result, LineKind.RESULT, message, answers=True))
        self.request_run()

    def notice(self, delegation_id: str | None, text: str) -> None:
        """Owe a fixed line, said word for word in its turn; an empty line says nothing."""
        if text.strip():
            self._owe(Owed(ReadyResult(delegation_id or "", self._own_id("notice"), text), LineKind.NOTICE))
            self.request_run()

    def ended(self, delegation_id: str) -> None:
        """Forget, in its turn, what the caller heard of a delegation that ended with nothing more to say."""
        self._owe(Turnover(delegation_id))

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """Move, in its turn, what the caller heard of a delegation to ``taker``, which carries its work."""
        self._owe(Turnover(delegation_id, taker))

    def commentary(self, text: str, delegation_id: str | None = None) -> None:
        """Drop every pending lead-in, then owe the application's commentary, unless blank, for ``delegation_id``."""
        self.acknowledgements.drop_all()
        if not text.strip():
            return
        self._worded(ReadyResult(delegation_id or "", self._own_id("commentary"), text), text)

    def greet(self, instructions: str) -> None:
        """Greet the caller before they speak: ``instructions`` are pinned and worded, else the greeting line."""
        self.instructions.append(instructions)
        self.version += 1
        self._worded(Instruction("", self._own_id("commentary"), instructions, self.lines.greeting), instructions)

    def _worded(self, said: ReadyResult, text: str) -> None:
        message = {"role": "developer", "content": self.prompts.context_label + text}
        self._register(message, APP_CONTEXT)
        self.unsaid.add(id(message))
        self._owe(Owed(said, LineKind.COMMENTARY, message))
        self.request_run()

    def thinking(self, text: str) -> None:
        """Drop every pending lead-in, then keep silent context, unless blank, for later decisions and delegations."""
        self.acknowledgements.drop_all()
        if not text.strip():
            return
        self.heard_log.context(text)
        self._append({"role": "developer", "content": self.prompts.context_label + text}, APP_CONTEXT, run=False)
        self._move()

    def typed(self, text: str) -> None:
        """Put the application's typed question into the talker's context, but not into the heard log."""
        self._append({"role": "developer", "content": self.prompts.context_label + text}, TYPED, run=False)

    async def instruction(self, text: str) -> None:
        """Pin a new direction, cut the line in progress, and decide it in turn; a blank one does nothing."""
        if not text.strip():
            return
        self.instructions.append(text)
        self.heard_log.instruction(text)
        self.instruction_due = True
        self._move()
        if self.talker is not None:
            # An interruption at the talker, not at the pipeline's start: the caller's turn in progress stays.
            await self.talker.queue_frame(InterruptionFrame())
        self.request_run()

    def attach(
        self,
        user: LLMUserAggregator,
        assistant: LLMAssistantAggregator,
        worker: PipelineWorker,
    ) -> None:
        """Hear each assistant turn's end, the caller's idle spells, and the talker's errors in the pipeline."""

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message):
            if message.interrupted:
                self.cut_marked(aggregator.context, message.content)
            elif line := self.spoken.end_played(message.content):
                self.heard_log.partly(line.number, line.heard)
                self._moved()
            self.bound()

        @user.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(aggregator):
            self.on_user_idle()

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            if frame.processor is self.talker:
                self.recover()

    def on_user_idle(self) -> None:
        """Run whatever is still due once the caller has been idle: a turn with no words brings no run."""
        if self._due():
            self.run_requested = False
            self.request_run()

    def cut_marked(self, context: LLMContext, played: str) -> None:
        """Settle the cut line with the words the aggregator recorded, ``played``, and cut-mark its context record."""
        kept = self.spoken.cut_played(played)
        if not kept:
            return
        if not played:
            context.add_message({"role": "assistant", "content": kept.rstrip() + self.prompts.cut_mark})
        else:
            message = next((each for each in reversed(context.get_messages()) if self.kind_of(each) is SPOKEN), None)
            if message is None or standard_message_text(message) != played:
                return
            message["content"] = kept.rstrip() + self.prompts.cut_mark
        self._move()

    async def _quiet(self, after: float) -> None:
        """Offer a progress line when work has run through a quiet spell of ``after`` seconds."""
        tick = max(0.05, min(1.0, after / 4))
        cap = self.config.delegation.progress_max_per_delegation
        while True:
            await asyncio.sleep(tick)
            self._moved()
            if not self.floor.idle or self.progress_due is not None:
                continue
            entry = self.worknote.quiet_candidate(cap)
            if entry is None:
                continue
            if self.clock() - max(self.floor.idle_since, entry.created_at, self._offered_at) < after:
                continue
            self._offered_at = self.clock()
            self.progress_due = entry
            self.request_run()
