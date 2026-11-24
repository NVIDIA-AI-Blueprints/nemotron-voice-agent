# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's voice: which run of the talker each one is, what its decision does, and what the backend sends."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from hashlib import sha256
from typing import TYPE_CHECKING, ClassVar
from weakref import WeakKeyDictionary

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextUserTurnFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
)
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext, standard_message_text

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import DelegationReader, ReadyResult
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.display import NullDisplay
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.handoffs import Acknowledgements, HandOffs
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.heard_log import HeardLog
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.models.decision import (
    ACTIONS,
    UNUSABLE,
    CallSpec,
    TalkerDecision,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind, Outcome, SpokenLines
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import (
    CALL_FIELDS,
    RESPONSE_KEY,
    Layout,
    TalkerCall,
    after_failure,
    floored,
    relay_line,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import Attribution
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.worknote import ProgressKey, ProgressSource, WorkNote
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens
from nemotron_voice_agent.pipecat.shared.observe import log_observation

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator, LLMUserAggregator

    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.config_manager.schema import LiveConfig
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.display import DisplaySink
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import FixedLines, PromptSet
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.talker import Request, ResponseFormat, TalkerChat
    from nemotron_voice_agent.pipecat.runtime.gateways.live.ports import DelegationPort
    from nemotron_voice_agent.pipecat.shared.delegation_contract import SettleCallerTurn


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
class Greeting(ReadyResult):
    """The one-time greeting instructions and the fixed greeting used if wording fails."""

    fallback: str

    @property
    def unworded(self) -> str:
        """The fallback line."""
        return self.fallback


@dataclass(frozen=True)
class NamedNotice(ReadyResult):
    """A notice whose writer sees its request, with the original fixed line as fallback."""

    fallback: str

    @property
    def unworded(self) -> str:
        """The original notice when its single wording call cannot supply a line."""
        return self.fallback


class RelayScope(Enum):
    """The context a Writer batch can read."""

    CONVERSATION = auto()
    WORK = auto()
    ROUTING_QUESTION = auto()


@dataclass(eq=False)
class Owed:
    """Something to say, with an optional reference message to retain alongside the spoken line."""

    said: ReadyResult
    kind: LineKind
    message: dict | None = None
    answers: bool = False

    @property
    def worded(self) -> bool:
        """Whether it is worded by the relay, rather than said as written."""
        return self.message is not None or isinstance(self.said, Greeting)


@dataclass(frozen=True)
class Turnover:
    """The end of a delegation, or its handover to ``taker``, applied in its turn."""

    delegation_id: str
    taker: str | None = None
    branch_ref: str | None = field(default=None, kw_only=True)


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
        routing_failed = decision.diagnostics.get("recovery") in UNUSABLE
        # Words already said answer the turn, but a ``delegate`` call made with them still hands it over.
        if call.said and not (decision.action == "delegate" and decision.call_id):
            decision = TalkerDecision("speak", call.said, decision.diagnostics)
        else:
            decision = floored(decision, work_open=self.work_open, holding=voice.lines.holding)
        voice.decided_by(self)
        voice.observe("talker.completed", action=decision.action, **decision.diagnostics)
        return voice._apply_routing(self, call, decision, routing_failed=routing_failed)

    def dropped(self, voice: Voice) -> None:
        """Leave its turns for the run the result brings, which decides them with the result in view."""
        voice.observe("talker.cut_off")

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Leave its turns undecided, unless its words went to speech."""
        if call.said:
            voice.decided_by(self)
            voice._settle_caller_turns(self.covers)

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Take its turns as decided; return whether the fallback is due: not when its words went to speech."""
        voice.decided_by(self)
        if call.said:
            voice._settle_caller_turns(self.covers, routing_failed=True)
        return not call.said

    def recover(self, voice: Voice) -> None:
        """Keep the failed turn at the frontend while work is open; otherwise delegate without a lead-in."""
        decision = after_failure({"recovery": "talker_failed"}, work_open=self.work_open, holding=voice.lines.holding)
        voice._apply_routing(self, None, decision, routing_failed=True)


@dataclass
class Relay:
    """A relay: what is owed, worded into one line with the writer's rules."""

    batch: tuple[Owed, ...]
    stamp: tuple

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Check for a compatible result once it enters the queue."""
        return False

    def moved(self, voice: Voice) -> bool:
        """Return whether the first batch or the evidence it uses changed."""
        batch = voice._owed_batch((*self.batch, *voice.owed))
        return batch != self.batch or voice._stamp(self.batch) != self.stamp

    def holds(self, delegation_id: str) -> bool:
        """Return whether it words an item of ``delegation_id``, whose end or handover then waits for its re-word."""
        return any(item.worded and item.said.delegation_id == delegation_id for item in self.batch)

    def delivered(self, voice: Voice, decision: TalkerDecision | None) -> Line | None:
        """Honor valid wording or omission; on failure consider raw fallback independently per item."""
        words = relay_line(decision)
        if words is None:
            return voice.relay_fallback(self.batch)
        if not words.strip():
            voice.delivered(tuple(item for item in self.batch if item.worded), "")
            fixed = tuple(item for item in self.batch if not item.worded)
            return voice.guard_presentation(voice.delivered(fixed, voice.joined(fixed, "")), fixed) if fixed else None
        return voice.guard_presentation(voice.delivered(self.batch, voice.joined(self.batch, words)), self.batch)

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Return the worded line, or the items as written when it fails the checks; nothing once it moved."""
        if self.moved(voice):
            self.dropped(voice)
            return ""
        delivered = self.delivered(voice, decision)
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
        """Recheck the current source frontier before any raw fallback; make no further model call."""
        line = voice.relay_fallback(self.batch, fixed_current=not self.moved(voice))
        if line is not None:
            voice.say(line)


@dataclass
class Progress:
    """One progress line from a frozen batch of quiet jobs' backend evidence."""

    sources: tuple[ProgressSource, ...]
    version: int

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Return ``True``: a result moves what the line is about."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Return whether a new item or the voice's version moved since it was offered."""
        return (
            voice.version != self.version
            or any(isinstance(entry, Owed) for entry in voice.owed)
            or not voice.worknote.progress_current(self.sources)
        )

    def holds(self, delegation_id: str) -> bool:
        """Return ``False``: it words no item."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Say the line only if nothing moved while it was worded; its news is taken either way then."""
        ids = tuple(dict.fromkeys(source.id for source in self.sources))
        if self.moved(voice):
            voice.observe("progress.dropped", delegations=list(ids))
            return ""
        voice.worknote.progress_worded(self.sources)
        text = (relay_line(decision) or "").strip()
        refs = frozenset(source.branch_ref for source in self.sources if source.branch_ref is not None)
        if not text:
            voice.observe("progress.silent", delegations=list(ids))
            return ""
        if len(self.sources) == 1:
            source = self.sources[0]
            request = voice.worknote.entries.get(source.id)
            if request is not None and voice.spoken.heard_handoff(text, source.id, source.branch_ref, request.parts):
                voice.observe("progress.repeated_handoff", delegation_id=source.id)
                return ""
        voice.observe("progress.said", delegations=list(ids))
        return voice.hand(Line(LineKind.STATUS, text, ids, tells=ids, again=False, work_refs=refs, work_tells=refs))

    def dropped(self, voice: Voice) -> None:
        """Keep its news for the next quiet spell."""
        voice.observe("progress.dropped", delegations=[source.id for source in self.sources])

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
    """A relay's request and decision while it is worded out of band."""

    plan: Relay
    messages: list[dict]
    task: asyncio.Task | None = None
    decision: TalkerDecision | None = None
    completed: bool = False

    def cut(self) -> None:
        """Stop the wording if it is still under way."""
        if self.task is not None:
            self.task.cancel()


@dataclass(frozen=True)
class PresentationGuard:
    """A queued line's original items and context; never retained as an instruction to replay speech."""

    batch: tuple[Owed, ...]
    stamp: tuple
    input_stamp: tuple
    fallback_reason: str | None = None
    handed: bool = False


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
        self.spoken = SpokenLines(self, observe=observe)
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
        self._pending_caller_turns: dict[int, SettleCallerTurn] = {}
        self.input_delivery: dict[int, dict] = {}
        self.noted: dict[int, tuple[dict, int | None]] = {}
        self.unsaid: set[int] = set()
        self.sent: list[dict] = []
        self._writes: list[dict] = []
        self.owed: deque[Owed | Turnover] = deque()
        self.early: EarlyLine | None = None
        self._wording_changed: asyncio.Future[None] | None = None
        self.waiting_lines: deque[Line] = deque()
        self._presentation_guards: WeakKeyDictionary[Line, PresentationGuard] = WeakKeyDictionary()
        self.lead_ins: dict[str, Line] = {}
        self.instruction_due = False
        self.progress_due: tuple[str | ProgressKey, ...] | None = None
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
        self._attachment_snapshot: dict | None = None
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
        for settle in self._pending_caller_turns.values():
            settle()
        self._pending_caller_turns.clear()
        if self.early is not None:
            self.early.cut()
        tasks = [*self._tasks, *([self._quiet_task] if self._quiet_task else [])]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.owed.clear()
        self.waiting_lines.clear()
        self._presentation_guards.clear()

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

    def export_attachment(self, max_items: int) -> tuple[dict, Iterator[tuple[int, dict]]]:
        """Collect private reconnect evidence without forwarding work or speech."""
        flights = list(self.spoken.in_flight_lines())
        in_flight = {line.number for line in flights}
        pending = [
            *self.owed,
            *getattr(getattr(self.call, "plan", None), "batch", ()),
            *(item for line in self.waiting_lines for item in line.items),
        ]
        uncertain = dict.fromkeys(
            [
                *(line.text for line in flights if line.text),
                *(item.said.speak for item in pending if hasattr(item, "said") and item.said.speak),
                *(line.text for line in self.waiting_lines if line.text),
            ]
        )
        if self.context is not None:
            self._settle_results(self.context)
        messages = []
        for message in self._read():
            kind = self.kind_of(message)
            identity = id(message)
            noted = self.noted.get(identity)
            messages.append(
                {
                    "content": standard_message_text(message),
                    "kind": kind.name,
                    "decided": identity in self.decided,
                    "noted": noted is not None,
                    **({"caller_note_seq": noted[1]} if noted is not None and noted[1] is not None else {}),
                    "delivery": self.input_delivery.get(identity),
                    "call_fields": {key: message[key] for key in CALL_FIELDS if key in message},
                }
            )
        playback, rows = self.spoken.export_attachment(flights)
        return {
            "response_number": max(
                [self.responses, *in_flight, *(entry.line or 0 for entry in self.heard_log.entries)]
            ),
            "instructions": list(self.instructions),
            "instruction_due": self.instruction_due,
            "seed_roles": list(self.seed_roles),
            "messages": messages,
            "uncertain": list(uncertain)[-max_items:],
            "progress": self.worknote.attachment_progress(),
            **self.heard_log.export_attachment(in_flight, max_items),
            **playback,
        }, rows

    def restore_attachment(self, snapshot: dict) -> None:
        """Restore context behind the attachment fence, without rerunning caller requests."""
        if self.call is not None or self.spoken.in_flight() or self.owed:
            raise ValueError("Frontend attachment restore requires a fresh voice")
        response_number = snapshot.get("response_number", 0)
        if type(response_number) is not int or response_number < 0:
            raise ValueError("Invalid frontend attachment response number")
        kinds = {kind.name: kind for kind in (CALLER, SPOKEN, CALL, CALL_RESULT, RESULT, APP_CONTEXT, TYPED)}
        self.instructions = list(snapshot["instructions"])
        self.seed_roles = list(snapshot["seed_roles"])
        messages = []
        for item in snapshot["messages"]:
            kind = kinds[item["kind"]]
            message = {"role": kind.role, "content": item["content"], **item.get("call_fields", {})}
            self._register(message, kind)
            messages.append(message)
            delivery = item.get("delivery")
            if delivery is not None:
                self.input_delivery[id(message)] = dict(delivery)
            pending = not item.get("decided", True) or delivery and delivery.get("state") == "never_forwarded"
            if not pending:
                self.decided[id(message)] = message
            if item.get("noted", False):
                sequence = item.get("caller_note_seq")
                self.noted[id(message)] = (message, sequence if type(sequence) is int else None)
        if snapshot.get("messages_omitted"):
            omitted = {"role": APP_CONTEXT.role, "content": self.heard_log.words.omitted}
            if not any(self.kind_of(message) is APP_CONTEXT and message == omitted for message in messages):
                self._register(omitted, APP_CONTEXT)
                messages.insert(0, omitted)
        if snapshot["uncertain"]:
            note = self.prompts.reconnect_note().format(text="\n".join(snapshot["uncertain"]))
            message = {"role": "developer", "content": self.prompts.context_label + note}
            self._register(message, APP_CONTEXT)
            messages.append(message)
        uncertain_inputs = [
            row
            for row in snapshot["messages"]
            if row.get("delivery") and row["delivery"].get("state") in {"unknown", "attempted"}
        ]
        if uncertain_inputs:
            message = {"role": "developer", "content": json.dumps({"private_input_delivery": uncertain_inputs})}
            self._register(message, APP_CONTEXT)
            messages.append(message)
        present_results = {message.get("tool_call_id") for message in messages}
        paired = []
        for message in messages:
            paired.append(message)
            for call in message.get("tool_calls", ()):
                if call["id"] not in present_results:
                    placeholder = {"role": "tool", "tool_call_id": call["id"], "content": '{"status": "unknown"}'}
                    self._register(placeholder, CALL_RESULT)
                    paired.append(placeholder)
        self.context.set_messages(paired)
        self.instruction_due = bool(snapshot.get("instruction_due", False))
        self.heard_log.restore_attachment(snapshot)
        self.responses = max([response_number, *(entry.line or 0 for entry in self.heard_log.entries)])
        if snapshot["uncertain"]:
            self.heard_log.uncertain_after_reconnect()
        self._attachment_snapshot = snapshot

    def resume_attachment(self) -> None:
        """Restore scoped presentation evidence after replay reconstructs work ownership."""
        snapshot = self._attachment_snapshot
        if snapshot is None:
            return
        ownership = self.reader.ownership
        self.worknote.restore_request_wording(self._confirmed_requests(snapshot["messages"]), ownership)
        self.worknote.restore_attachment_progress(snapshot.get("progress", []))
        self.spoken.restore_attachment(snapshot, ownership.bindings)
        self._attachment_snapshot = None

    @staticmethod
    def _confirmed_requests(messages: list[dict]) -> dict[str, str]:
        """Recover complete caller wording for accepted handoffs."""
        grouped: dict[str, list[dict]] = {}
        for row in messages:
            delivery = row.get("delivery") or {}
            source = delivery.get("delegation_id")
            if row["kind"] == "caller" and delivery.get("state") == "accepted" and isinstance(source, str) and source:
                grouped.setdefault(source, []).append(row)
        confirmed = {}
        for source, parts in grouped.items():
            wording = " ".join(row["content"] for row in parts)
            digest = sha256(wording.encode()).hexdigest()
            if all(row["delivery"].get("request_digest") == digest for row in parts):
                confirmed[source] = wording
        return confirmed

    def _read(self, retained: Sequence[dict] | None = None) -> list[dict]:
        """Return what the calls read: the context and the voice's appends on their way, but notes and unsaid items."""
        if retained is None:
            retained = self.context.get_messages() if self.context else []
        messages = list(retained)
        held = {id(message) for message in messages}
        on_the_way = [message for message in self.sent if id(message) not in held]
        return [m for m in [*messages, *on_the_way] if self.kind_of(m) is not None and id(m) not in self.unsaid]

    def bound(self, *, request: Request | None = None) -> bool:
        """Drop the oldest turn groups once the context passes ``memory.compact_at`` of ``memory.context_tokens``.

        The count holds the instructions and each line in flight whole; each caller turn opens a group, and the newest
        group never leaves. When given a request, compact against its rendered size and the full budget instead.
        """
        memory = self.config.memory
        if self.context is None:
            return request is None or sum(map(message_tokens, request.messages(self._read()))) <= memory.context_tokens
        limit = memory.context_tokens if request is not None else memory.compact_at * memory.context_tokens
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
        retained = messages
        size = tokens if request is None else sum(map(message_tokens, request.messages(self._read(retained))))
        for group, weight in zip(groups[:-1], weights[:-1], strict=True):
            if size <= limit:
                break
            protected = False
            for index in group:
                message = messages[index]
                kind = self.kind_of(message)
                delivery = self.input_delivery.get(id(message), {})
                if (kind is not None and kind.turn and id(message) not in self.decided) or delivery.get("state") in {
                    "never_forwarded",
                    "unknown",
                    "attempted",
                }:
                    protected = True
                    break
            if protected:
                continue
            dropped.update(group)
            tokens -= weight
            if request is None:
                size = tokens
            else:
                retained = [message for index, message in enumerate(messages) if index not in dropped]
                size = sum(map(message_tokens, request.messages(self._read(retained))))
        if dropped and (request is None or size <= limit):
            for index in dropped:
                key = id(messages[index])
                for records in (
                    self.kinds,
                    self.decided,
                    self.noted,
                    self.input_delivery,
                ):
                    records.pop(key, None)
            if request is None:
                retained = [message for index, message in enumerate(messages) if index not in dropped]
            self.context.set_messages(retained)
        return size <= memory.context_tokens

    def _tidy(self) -> None:
        """Remove items not said yet and compact retained conversation history."""
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

    def hold_pending_caller_turns(self) -> None:
        """Register finalized, undecided input before routing it or reopening a restored call."""
        begin = getattr(self.delegation, "begin_caller_turn", None)
        if self._closed or not callable(begin):
            return
        for message in self._undecided():
            key = id(message)
            if key not in self._pending_caller_turns and (settle := begin()) is not None:
                self._pending_caller_turns[key] = settle

    def _settle_caller_turns(self, messages: Sequence[dict], *, routing_failed: bool = False) -> None:
        for message in messages:
            settle = self._pending_caller_turns.pop(id(message), None)
            if settle is not None:
                if routing_failed:
                    settle(routing_failed=True)
                else:
                    settle()

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
        if not (note or undecided or self.instruction_due or self.waiting_lines) and self.owed:
            try:
                await self._worded_early()
            except asyncio.CancelledError:
                self._after_call()
                raise
            undecided = self._undecided()
        if note or undecided or self.instruction_due:
            call = self._routing(undecided, note)
        elif not self.floor.caller_speaking:
            if self.waiting_lines:
                call = self.saying(self._next_line())
            elif self.owed:
                if batch := self._take():
                    call = self._said_by(batch)
            elif self.progress_due is not None:
                call = self._progress()
        self.call = call
        if call is None:
            self._after_call()
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

    def _routing(self, undecided: list[dict], note: str = "") -> TalkerCall:
        """Return the routing call on the caller's undecided turns, an instruction or an empty turn's ``note``."""
        turn = " ".join(standard_message_text(message) for message in undecided)
        request = turn or (self.instructions[-1] if self.instruction_due else "")
        plan = Routing(tuple(undecided), request, bool(self.worknote.open()))
        self._tidy()
        routing = self.layout.routing("\n\n".join(self.instructions), self.worknote.line(), turn, note)
        if not self.bound(request=routing):
            if undecided:
                caller_notes = set()
                for message in undecided:
                    key = id(message)
                    if (noted := self.noted.get(key)) is not None and noted[1] is not None:
                        caller_notes.add(noted[1])
                    if self.input_delivery.get(key, {}).get("state") == "never_forwarded":
                        self.input_delivery.pop(key)
                self.heard_log.routing_failed("context_budget", caller_notes)
            self.decided_by(plan)
            self._settle_caller_turns(plan.covers, routing_failed=True)
            self.observe("talker.routing_failed", reason="context_budget")
            return self.saying(Line(LineKind.REPLY, self.lines.apology))
        for message in undecided:
            if id(message) not in self.noted:
                sequence = self.heard_log.caller(standard_message_text(message))
                self.noted[id(message)] = (message, sequence)
        return TalkerCall(
            spec=self.routing_spec,
            request=routing,
            streams=plan.streams,
            format=self.routing_format,
            plan=plan,
        )

    def _owed_batch(self, entries: Iterable[Owed | Turnover]) -> tuple[Owed, ...]:
        """Group consecutive items that can use the same Writer context."""
        batch = []
        scope = None
        for item in entries:
            if not isinstance(item, Owed):
                continue
            if item.worded:
                item_scope = self._item_relay_scope(item)
                if scope is not None and scope != item_scope:
                    break
                scope = item_scope
            batch.append(item)
        return tuple(batch)

    def _take(self) -> list[Owed]:
        """Take one Writer batch and handle turnovers in queue order."""
        batch: list[Owed] = []
        count = len(self._owed_batch(self.owed))
        while self.owed:
            if len(batch) == count and isinstance(self.owed[0], Owed):
                break
            entry = self.owed.popleft()
            if isinstance(entry, Owed):
                batch.append(entry)
            elif not any(self._turnover_holds(entry, item) for item in batch):
                self._turnover(entry)
        return batch

    @staticmethod
    def _turnover_holds(entry: Turnover, item: Owed) -> bool:
        attribution = item.said.attribution
        return item.worded and (
            attribution is not None and attribution.branch_ref == entry.branch_ref
            if entry.branch_ref is not None
            else item.said.delegation_id == entry.delegation_id
        )

    def _turnover(self, entry: Turnover) -> None:
        if entry.branch_ref is not None:
            self.spoken.forget_work(entry.branch_ref)
        elif entry.taker is None:
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
            if early.completed and early.plan.stamp == self._stamp(batch):
                self.observe("answer.worded_early", items=len(batch))
                line = early.plan.delivered(self, early.decision)
                return self.saying(line) if line else None
        if prepared := self._prepare_relay(batch):
            call, _ = prepared
            return call
        line = self.relay_fallback(batch, reason="context_budget")
        return self.saying(line) if line else None

    def _prepare_relay(self, batch: Sequence[Owed]) -> tuple[TalkerCall, list[dict]] | None:
        """Budget the rendered Writer request, compacting only the history it reads."""
        scope = self._relay_scope(batch)
        with_history = scope is not RelayScope.ROUTING_QUESTION
        if with_history:
            self._tidy()
        history = self._read() if with_history else []
        call = self._relay_call(batch, history)
        limit = self.config.memory.context_tokens
        if sum(map(message_tokens, call.request.messages(history))) > limit:
            if not with_history or not self.bound(request=call.request):
                return None
            history = self._read()
            call = self._relay_call(batch, history)
            if sum(map(message_tokens, call.request.messages(history))) > limit:
                return None
        return call, [dict(message) for message in history]

    def _relay_call(self, batch: Sequence[Owed], history: Sequence[dict] = ()) -> TalkerCall:
        """Return the relay that words ``batch`` with its frozen source/presentation note."""
        routing_question = self._relay_scope(batch) is RelayScope.ROUTING_QUESTION
        content, note = self._update(batch)
        turn = next(
            (
                text
                for message in reversed(history)
                if self.kind_of(message) is CALLER and (text := standard_message_text(message)).strip()
            ),
            "",
        )
        instructions = "\n\n".join(self.instructions)
        relay = self.layout.relay(instructions, content, note, turn=turn, routing_question=routing_question)
        if routing_question:
            relay = replace(relay, include_history=False)
        elif turn and sum(map(message_tokens, relay.messages(history))) > self.config.memory.context_tokens:
            relay = self.layout.relay(instructions, content, note)
        plan = Relay(tuple(batch), self._stamp(batch))
        return TalkerCall(spec=self.answer_spec, request=relay, format=self.talker.response_format, plan=plan)

    @staticmethod
    def _update_scopes(batch: Sequence[Owed]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        worded = [item for item in batch if item.worded]
        legacy = tuple(
            dict.fromkeys(
                item.said.delegation_id for item in worded if item.said.delegation_id and item.said.attribution is None
            )
        )
        branches = tuple(
            dict.fromkeys(item.said.attribution.branch_ref for item in worded if item.said.attribution is not None)
        )
        return legacy, branches

    def _item_relay_scope(self, item: Owed) -> RelayScope:
        if item.said.routing_question and self.prompts.routing_question:
            return RelayScope.ROUTING_QUESTION
        if item.said.attribution is not None and item.said.attribution.focused:
            return RelayScope.WORK
        return RelayScope.CONVERSATION

    def _relay_scope(self, batch: Sequence[Owed]) -> RelayScope:
        """Read the scope shared by the batch's worded items."""
        return next((self._item_relay_scope(item) for item in batch if item.worded), RelayScope.CONVERSATION)

    def _question_context(self, sources: tuple[str, ...]) -> tuple[dict, ...]:
        """Keep the question's caller input and later caller turns, including turns handled locally."""
        caller = [
            (self.input_delivery.get(id(message), {}).get("delegation_id"), standard_message_text(message))
            for message in self._read()
            if self.kind_of(message) is CALLER
        ]
        rows = []
        for source in sources:
            positions = [index for index, (delegation_id, _) in enumerate(caller) if delegation_id == source]
            if positions:
                wording = [caller[index][1] for index in positions]
            elif entry := self.worknote.entries.get(source):
                wording = [words for _, words in entry.parts]
            else:
                wording = []
            context = {"delegation_id": source, "caller_requests": wording}
            if positions:
                context["later_caller_turns"] = [words for _, words in caller[positions[-1] + 1 :]]
            elif caller:
                context["latest_caller_turn"] = caller[-1][1]
            if wording or caller:
                rows.append(context)
        return tuple(rows)

    def _update(self, batch: Sequence[Owed]) -> tuple[str, str]:
        """Return result text with scoped caller input and truthful frontend presentation evidence."""
        legacy, branches = self._update_scopes(batch)
        told = self.spoken.context_text(legacy, branches)
        scope = self._relay_scope(batch)
        # A source identity binds a whole caller turn. It cannot identify the clause for this result.
        if scope is RelayScope.ROUTING_QUESTION:
            caller_context = self._question_context(legacy)
        else:
            caller_context = () if scope is RelayScope.WORK else self.worknote.newer_requests(legacy, branches)
        note = self.layout.note(told, caller_context)
        worded = [item.said for item in batch if item.worded]
        if len(worded) > 1 and (branches or len(legacy) > 1):
            updates = [
                {
                    "item_id": result.item_id,
                    "delegation_id": result.delegation_id or None,
                    "branch_ref": result.attribution.branch_ref if result.attribution is not None else None,
                    "text": result.speak,
                }
                for result in worded
            ]
            return json.dumps(updates, ensure_ascii=False), note
        return "\n\n".join(result.speak for result in worded), note

    def _stamp(self, batch: Sequence[Owed]) -> tuple:
        """Keep normal playback advancement from spending another wording call on unchanged facts."""
        legacy, branches = self._update_scopes(batch)
        return (
            tuple(batch),
            self.spoken.context_stamp(legacy, branches),
            self.version,
            self._input_stamp(batch),
        )

    def _input_stamp(self, batch: Sequence[Owed]) -> tuple:
        """Visible caller/app text and source relationships, independent of native bookkeeping/playback.

        Writer requests omit native call/result messages and join adjacent user/developer text. Their
        Python identities and placeholder replacement are not new presentation input. Freeze the visible
        text itself so an actual edit still invalidates a pending candidate.
        """
        legacy, branches = self._update_scopes(batch)
        if self._relay_scope(batch) is RelayScope.ROUTING_QUESTION:
            return tuple(self.instructions), self._question_context(legacy)
        return (
            tuple(self.instructions),
            self.worknote.newer_requests(legacy, branches),
            "\n".join(
                standard_message_text(message)
                for message in self._read()
                if (kind := self.kind_of(message)) and (kind.turn or kind in (APP_CONTEXT, TYPED))
            ),
        )

    def _deciding(self) -> bool:
        """Return whether a caller turn or an instruction waits for its routing call, or is being decided."""
        return self.instruction_due or bool(self._undecided())

    def _word_early(self) -> None:
        """Word what is owed out of band. Replace it when the caller is silent and the actual request changed."""
        if self.talker is None or self._closed:
            return
        batch = self._owed_batch(self.owed)
        early = self.early
        if not any(item.worded for item in batch):
            return
        if early is not None and (
            self.floor.caller_speaking or self._deciding() or early.plan.stamp == self._stamp(batch)
        ):
            return
        if self.waiting_lines:
            return
        prepared = self._prepare_relay(batch)
        if prepared is None:
            return
        call, history = prepared
        messages = call.request.messages(history)
        if early is not None:
            if early.plan.batch == call.plan.batch and early.messages == messages:
                early.plan.stamp = call.plan.stamp
                return
            early.cut()
        self.early = EarlyLine(call.plan, messages)
        words = self._words(self.early, call, LLMContext(history))
        self.early.task = asyncio.get_running_loop().create_task(words, name="early-wording")

    async def _words(self, early: EarlyLine, call: TalkerCall, context: LLMContext) -> None:
        """Retain the Writer decision; fallback eligibility is rechecked only at actual handoff."""
        try:
            async with asyncio.timeout(self.config.reliability.talker_attempt_timeout_seconds):
                early.decision = await self.talker.run_call(call, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Wording early failed ({type(exc).__name__}: {exc}); checking presentation fallback")
            self.observe("commentary.recovered", error_type=type(exc).__name__, results=len(early.plan.batch))
        early.completed = True

    async def _worded_early(self) -> None:
        """Wait for the wording under way of what is owed now, so the relay says its line rather than word it again."""
        self._word_early()
        while (early := self.early) is not None and early.task is not None and not early.task.done():
            if self.floor.caller_speaking or self._deciding() or self.waiting_lines:
                return
            batch = self._owed_batch(self.owed)
            if early.plan.stamp != self._stamp(batch):
                return
            changed = self._wording_changed = asyncio.get_running_loop().create_future()
            try:
                await asyncio.wait((early.task, changed), return_when=asyncio.FIRST_COMPLETED)
            finally:
                self._wording_changed = None

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

    def _raw_current(self, item: Owed) -> bool:
        """Allow raw fallback only with a retained original source and no later unincorporated input.

        This is a conservative identity/frontier check, not a decision that newer work superseded this
        result. Unknown lineage forbids raw fallback; a newer independent item's own frontier can pass.
        """
        if not item.worded or isinstance(item.said, (Greeting, NamedNotice)):
            return True
        if not self.worknote.publication_current(item.said.delegation_id, item.said.attribution):
            return False
        messages = self._read()
        sources = [
            (index, self.input_delivery[id(message)])
            for index, message in enumerate(messages)
            if id(message) in self.input_delivery
            and item.said.delegation_id == self.input_delivery[id(message)].get("delegation_id")
        ]
        if not sources or any(row.get("instruction_revision") != len(self.instructions) for _, row in sources):
            return False
        source_calls = {row["call_id"] for _, row in sources if row.get("call_id")}
        # This receipt cannot consume caller input received during the tool call.
        source_call_positions = [
            index
            for index, message in enumerate(messages)
            if any(call.get("id") in source_calls for call in message.get("tool_calls", ()))
        ]
        frontier = max(index for index, _ in sources)
        for index, message in enumerate(messages[frontier + 1 :], frontier + 1):
            kind = self.kind_of(message)
            if kind is not None and (kind.turn or kind in (APP_CONTEXT, TYPED)):
                return False
            if (
                kind is CALL_RESULT
                and message.get("tool_call_id") not in source_calls
                and (not source_call_positions or index >= min(source_call_positions))
            ):
                return False
        return True

    def relay_fallback(
        self, batch: Sequence[Owed], *, fixed_current: bool = True, reason: str = "writer_failed"
    ) -> Line | None:
        """Keep current independent raw items eligible and journal every unsafe item as not conveyed."""
        safe, withheld = [], []
        for item in batch:
            current = self._raw_current(item)
            if item.worded and isinstance(item.said, (Greeting, NamedNotice)):
                current = fixed_current
            (safe if current else withheld).append(item)
        if withheld:
            self.delivered(withheld, "", omission_reason=reason)
        self.observe("commentary.fallback", reason=reason, raw_items=len(safe), withheld_items=len(withheld))
        if not safe:
            return None
        line = self.delivered(safe, self.joined(safe, as_written([item for item in safe if item.worded])))
        return self.guard_presentation(line, safe, fallback_reason=reason)

    def guard_presentation(
        self, line: Line | None, batch: Sequence[Owed], *, fallback_reason: str | None = None
    ) -> Line | None:
        """Keep the frozen wording frontier until actual handoff, without persisting a speech replay plan."""
        if line is not None:
            self._presentation_guards[line] = PresentationGuard(
                tuple(batch), self._stamp(batch), self._input_stamp(batch), fallback_reason
            )
        return line

    def delivered(self, batch: Sequence[Owed], text: str, *, omission_reason: str = "writer_omitted") -> Line | None:
        """Return the line that says ``batch``, its items said and what it told worded in; ``None`` with no words."""
        if not text:
            self._not_conveyed(batch, omission_reason)
            return None
        kinds = {item.kind for item in batch}
        kind = next(each for each in LINE_KIND_ORDER if each in kinds)
        ids = tuple(dict.fromkeys(item.said.delegation_id for item in batch if item.said.delegation_id))
        refs = frozenset(item.said.attribution.branch_ref for item in batch if item.said.attribution is not None)
        return Line(kind, text, ids, items=tuple(batch), results=any(item.answers for item in batch), work_refs=refs)

    def _not_conveyed(self, batch: Sequence[Owed], reason: str) -> None:
        """Settle each result and keep its original reference available for later recap."""
        evidence = [
            {
                "delegation_id": item.said.delegation_id,
                "item_id": item.said.item_id,
                "branch_ref": item.said.attribution.branch_ref if item.said.attribution is not None else None,
                "task_ref": item.said.attribution.task_ref if item.said.attribution is not None else None,
                "kind": item.kind.value,
            }
            for item in batch
            if not isinstance(item.said, Greeting)
        ]
        for item in batch:
            if item.message is not None and id(item.message) in self.unsaid:
                self.unsaid.remove(id(item.message))
                self._append(item.message, self.kind_of(item.message), run=False)
        if evidence:
            note = self.heard_log.not_conveyed(evidence, reason)
            self._append({"role": "developer", "content": note}, RESULT, run=False)
        self.observe("answer.not_conveyed", reason=reason, items=evidence)
        self._done(batch)

    def _progress(self) -> TalkerCall | None:
        """Return a progress line about the quiet work offered, worded from the work note, if it is still quiet."""
        ids, self.progress_due = self.progress_due, None
        sources, update = self.worknote.prepare_progress(ids, self.config.delegation.progress_max_per_delegation)
        if not sources:
            return None
        if any(source.branch_ref is not None for source in sources):
            told = self.spoken.context_text(
                tuple(source.id for source in sources if source.branch_ref is None),
                tuple(source.branch_ref for source in sources if source.branch_ref is not None),
            )
        else:
            told = "\n".join(filter(None, (self.spoken.told_text(source.id) for source in sources)))
        note = self.layout.note(told)
        self._tidy()
        progress = self.layout.progress("\n\n".join(self.instructions), update, note)
        if not self.bound(request=progress):
            return None
        self.worknote.progress_offered(sources)
        self.observe("progress.offered", delegations=[source.id for source in sources])
        plan = Progress(sources, self.version)
        return TalkerCall(spec=self.progress_spec, request=progress, format=self.talker.response_format, plan=plan)

    def speaking(self, text: str) -> None:
        """Add a routing ``speak``'s words to this response's line as they go to speech."""
        line = self.spoken.lines.get(self.responses)
        if line is None:
            self.hand(Line(LineKind.REPLY, text))
        else:
            line.text += text
            self.heard_log.replace_agent(line.number, line.text)

    def finished(self, call: TalkerCall, decision: TalkerDecision | None) -> str:
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

    def _apply_routing(
        self,
        plan: Routing,
        call: TalkerCall | None,
        decision: TalkerDecision,
        *,
        routing_failed: bool = False,
    ) -> str:
        if decision.action != "delegate" or not plan.request:
            self._settle_caller_turns(plan.covers, routing_failed=routing_failed)
        return self.actions[decision.action](plan, call, decision)

    def _delegated(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """Hand the turn to the delegation part with the decision's speech as its lead-in; with no request, nothing.

        Words of the call already said answer the turn as a reply, and the hand-off says no lead-in of its own.
        """
        lead_in, text = decision.speech, ""
        if call is not None and call.said:
            lead_in, text = "", self._spoke(plan, call, TalkerDecision("speak", call.said, decision.diagnostics))
        if plan.request:
            decided_at = self.floor.caller_starts
            frontier = {
                "state": "never_forwarded",
                "delegation_id": None,
                "instruction_revision": len(self.instructions),
                "call_id": decision.call_id,
                "request_digest": sha256(plan.request.encode()).hexdigest(),
            }
            for message in plan.covers:
                self.input_delivery[id(message)] = frontier

            def receipt(state: str, delegation_id: str | None) -> None:
                frontier.update(state=state, delegation_id=delegation_id)
                if state != "attempted":
                    self._settle_caller_turns(plan.covers)

            self.handoffs.delegate(
                plan.request,
                lead_in,
                decided_at=decided_at,
                call_id=decision.call_id,
                receipt=receipt,
            )
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
        """Validate the line against current input, then hand it to speech and record it as queued."""
        guard = self._presentation_guards.get(line)
        if guard is not None:
            current = (
                guard.input_stamp == self._input_stamp(guard.batch)
                if guard.handed
                else guard.stamp == self._stamp(guard.batch)
            )
            if guard.fallback_reason is not None and (
                not current or not all(self._raw_current(item) for item in guard.batch)
            ):
                self._presentation_guards.pop(line, None)
                refreshed = self.relay_fallback(guard.batch, fixed_current=current, reason=guard.fallback_reason)
                return self.hand(refreshed) if refreshed is not None else ""
            if guard.fallback_reason is None and not current:
                self._presentation_guards.pop(line, None)
                self.owe_again(guard.batch)
                self.observe("answer.wording_moved", items=len(guard.batch))
                self.request_run()
                return ""
        if guard is None or not guard.handed:
            for item in line.items:
                if item.message is not None:
                    if item.said.delegation_id:
                        if item.said.attribution is not None:
                            self.spoken.forget_work(item.said.attribution.branch_ref)
                        else:
                            self.spoken.forget(item.said.delegation_id)
                    self.unsaid.discard(id(item.message))
            if line.items:
                self.observe("answer.said", items=len(line.items), delegations=list(line.delegation_ids))
        if guard is not None:
            self._presentation_guards[line] = replace(guard, handed=True)
        said_as, line.number = line.number, self.responses
        self.spoken.handed(line)
        self.heard_log.presented(line, self.spoken.presentation(line), replaces=said_as if line.retried else None)
        self._write_with(line)
        self.bound()
        self._move()
        return line.text

    def _write_with(self, line: Line) -> None:
        """Put the items ``line`` says on their way into the context, out of where a line nobody heard left them."""
        self._writes = [item.message for item in line.items if item.message is not None]
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

    def interrupted(self) -> int:
        """Drop speech retries and return the latest response to which the interruption applies."""
        newest = self.responses
        self._drop_retries()
        return newest

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
        if self.floor.caller_speaking:
            self.observe("caller.turn.ended", turn=self.floor.caller_starts)
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
        if self._wording_changed is not None and not self._wording_changed.done():
            self._wording_changed.set_result(None)

    def _move(self) -> None:
        """Note that something every wording reads moved: a line said or taken back, an instruction, thinking, a cut."""
        self.version += 1
        self._moved()

    def audio_started(self, line: Line) -> None:
        """Attribute the frontend's observed audio start without inferring missing words."""
        self.heard_log.presented(line, self.spoken.presentation(line))
        self.observe(
            "line.audio_started",
            line=line.number,
            line_kind=line.kind.value,
            after_turn=self.floor.caller_starts if not self.floor.caller_speaking else None,
            result_delegations=list(
                dict.fromkeys(
                    item.said.delegation_id for item in line.items if item.answers and item.said.delegation_id
                )
            ),
        )

    def heard(self, line: Line) -> None:
        """Settle observed playout, retaining confirmed words even if synthesis failed."""
        failed = line.outcome is Outcome.SYNTHESIS_FAILED
        self.heard_log.presented(line, self.spoken.presentation(line))
        self._presentation_guards.pop(line, None)
        if failed and line.items:
            self._not_conveyed(line.items, "synthesis_failed")
        else:
            self._done(line.items)
        self._moved()

    def not_heard(self, line: Line, *, audio_exhausted: bool = False) -> None:
        """Return interrupted items to the queue or settle silent items after their audio attempts."""
        if line.items and line.number and not audio_exhausted and not line.audio_started:
            self.heard_log.presented(line, self.spoken.presentation(line))
        else:
            self.heard_log.unsay(line.number)
        self._presentation_guards.pop(line, None)
        if line.items:
            if audio_exhausted:
                self._not_conveyed(line.items, "no_audio")
            else:
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
            if item.message is not None:
                self.unsaid.add(id(item.message))
        self._take_out({id(item.message) for item in items if item.message is not None})

    def _done(self, items: Sequence[Owed]) -> None:
        """Count each result among ``items`` as said in the work note."""
        for item in items:
            if item.answers and item.said.delegation_id:
                self.worknote.said(
                    item.said.delegation_id,
                    attribution=item.said.attribution,
                    item_id=item.said.item_id,
                )

    def say(self, line: Line) -> None:
        """Say ``line`` with no model call, in its turn once nobody speaks."""
        if line not in self._presentation_guards:
            self.spoken.register(line)
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
        request = self.worknote.entries.get(delegation_id)
        line = Line(
            LineKind.LEAD_IN,
            text,
            (delegation_id,),
            tells=(delegation_id,),
            handoff_request=request.parts if request is not None else (),
        )
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
        while self.owed and isinstance(head := self.owed[0], Turnover) and not self._holds_turnover(head):
            self._turnover(self.owed.popleft())
        self._moved()

    def _holds_turnover(self, entry: Turnover) -> bool:
        if entry.branch_ref is None:
            return self._holds(entry.delegation_id)
        return (
            self.call is not None
            and isinstance(self.call.plan, Relay)
            and any(self._turnover_holds(entry, item) for item in self.call.plan.batch)
        )

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
        """Word a task-bound notice once, retaining its fixed line as the writer's fallback."""
        self._notice(delegation_id, text)

    def work_notice(self, delegation_id: str, text: str, attribution: Attribution) -> None:
        """Report only this branch; a combined parent request cannot supply a guessed subject."""
        self._notice(delegation_id, text, attribution=attribution)

    def _notice(self, delegation_id: str | None, text: str, *, attribution: Attribution | None = None) -> None:
        if text.strip():
            entry = (
                self.worknote.branches.get(attribution.branch_ref)
                if attribution is not None
                else self.worknote.entries.get(delegation_id)
            )
            if attribution is not None and entry is not None and not entry.request_known:
                entry = None
            message = None
            said = ReadyResult(delegation_id or "", self._own_id("notice"), text, attribution=attribution)
            if entry is not None:
                content = self.prompts.notice_update.format(request=entry.request, text=text)
                message = {"role": "developer", "content": self.prompts.context_label + content}
                self._register(message, APP_CONTEXT)
                self.unsaid.add(id(message))
                said = NamedNotice(delegation_id, said.item_id, content, text, attribution=attribution)
            self._owe(Owed(said, LineKind.NOTICE, message))
            self.request_run()

    def work_binding(self, branch_ref: str, delegation_id: str) -> None:
        """Associate a host-observed branch with actual unscoped lead-ins, without retagging their identities."""
        branch = self.worknote.branches.get(branch_ref)
        if branch is not None:
            for predecessor in branch.predecessor_refs:
                self.spoken.inherit_work(branch_ref, predecessor)
        self.spoken.bind_work(branch_ref, delegation_id)

    def work_ended(self, branch_ref: str, delegation_id: str) -> None:
        """End only this branch; defer consuming its spoken context behind its own queued result."""
        branch = self.worknote.branches.get(branch_ref)
        if branch is not None and branch.id != delegation_id:
            return
        self.worknote.branch_ended(branch_ref)
        self._owe(Turnover(delegation_id, branch_ref=branch_ref))

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
        """Ask the Writer for the startup greeting without retaining its instructions."""
        greeting = Greeting("", self._own_id("commentary"), instructions, self.lines.greeting)
        self._owe(Owed(greeting, LineKind.COMMENTARY))
        self.request_run()

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
        playout_frame: ContextVar[Frame | None] = ContextVar("playout_frame", default=None)

        @user.event_handler("on_before_push_frame")
        async def before_caller_turn(_, frame):
            if isinstance(frame, LLMContextUserTurnFrame) and not frame.speculation:
                self.hold_pending_caller_turns()
                self._moved()

        @assistant.event_handler("on_before_process_frame")
        async def identify_playout(_, frame):
            # Pipecat's turn callback can run after another frame starts. Its task inherits this frame's identity.
            playout_frame.set(frame if isinstance(frame, (InterruptionFrame, LLMFullResponseEndFrame)) else None)

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message):
            frame = playout_frame.get()
            if isinstance(frame, InterruptionFrame) and message.interrupted:
                self.cut_marked(aggregator.context, frame.id, message.content)
            elif (
                isinstance(frame, LLMFullResponseEndFrame) and (number := frame.metadata.get(RESPONSE_KEY)) is not None
            ):
                self.spoken.end_played(number, message.content)
            self.bound()

        @assistant.event_handler("on_after_process_frame")
        async def settle_playout_without_transcript(_, frame):
            # Run after any turn-stop callback queued by this frame. A settled receipt is untouched.
            if isinstance(frame, InterruptionFrame):
                asyncio.get_running_loop().call_soon(self.spoken.cut_played, frame.id, "")
            elif (
                isinstance(frame, LLMFullResponseEndFrame) and (number := frame.metadata.get(RESPONSE_KEY)) is not None
            ):
                asyncio.get_running_loop().call_soon(self.spoken.end_played, number, "")

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

    def cut_marked(self, context: LLMContext, interruption: int, played: str) -> None:
        """Settle the cut line with the words the aggregator recorded, ``played``, and cut-mark its context record."""
        if not self.spoken.cut_played(interruption, played) or not played:
            return
        message = next((each for each in reversed(context.get_messages()) if self.kind_of(each) is SPOKEN), None)
        if message is None or standard_message_text(message) != played:
            return
        message["content"] = played.rstrip() + self.prompts.cut_mark
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
            entries = tuple(
                entry for entry in self.worknote.quiet_candidates(cap) if self.clock() - entry.created_at >= after
            )
            if not entries:
                continue
            if self.clock() - max(self.floor.idle_since, self._offered_at) < after:
                continue
            self._offered_at = self.clock()
            self.progress_due = tuple(self.worknote.progress_key(entry) for entry in entries)
            self.request_run()
