# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's voice: which run of the talker each one is, what its decision does, and what the backend sends.

How backend answers land (``answers``): a cut answer's unheard sentences are said once, as written, only when the first
decision after the cut waits (``unheard_rest_once``). A line nobody heard is said again with its words while nothing
newer changed it, at most ``max_tries`` times for a backend answer, which is then dropped and the backend told
(``requeued_as_said``). A lead-in is skipped in front of an answer that is due, beside another lead-in, or when it
repeats the last line (``lead_in_skip``).
"""

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
from pipecat.utils.text.alnum_utils import alnum_only

from examples.frontend_backend_live.delegations import DelegationReader, ReadyResult
from examples.frontend_backend_live.display import NullDisplay
from examples.frontend_backend_live.handoffs import Acknowledgements, HandOffs
from examples.frontend_backend_live.heard_log import HEARD, HeardLog, Unheard, UnheardHow
from examples.frontend_backend_live.models.decision import ACTIONS, CallSpec, TalkerDecision, decision_from_output_text
from examples.frontend_backend_live.spoken import Line, LineKind, Outcome, SpokenLines, sentences, unheard_rest
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
from examples.shared.delegation_contract import ReplyUnheard, ends_with_question
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
    # How often a line that carried it settled not heard.
    tries: int = 0
    # The text of the line that last carried it; a re-queued item is said again with these words.
    words: str = ""
    # The newer requests when its line was said (WorkNote.newer_state of its delegations). A re-say holds only while
    # this is the same, so a request that opened and was already answered by the next run still counts as a change.
    # The update's "told" part is left out on purpose: delivered() forgets it, so it is gone when a line comes back.
    newer: tuple = ()
    # The line that last carried it produced no audio, even when said once more.
    silent: bool = False

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
    # A run on Pipecat's empty-turn note alone: a caller turn with no words.
    wordless: bool = False
    # It decides an application instruction.
    steered: bool = False

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
        if decision.action != "delegate":
            voice.turn_pending = None  # no delegation sends this turn's snapshot
        voice.decided_by(self)
        voice.observe("talker.completed", action=decision.action, **decision.diagnostics)
        said = voice.actions[decision.action](self, call, decision)
        voice._rest_after(self, decision)  # after the action: a reply is handed first
        return said

    def dropped(self, voice: Voice) -> None:
        """Leave its turns for the run the result brings, which decides them with the result in view."""
        voice.observe("talker.cut_off")

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Leave its turns undecided, unless its words went to speech."""
        if call.said:
            voice.decided_by(self)
            voice._rest_unruled()

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Take its turns as decided; return whether the fallback is due: not when its words went to speech."""
        voice.decided_by(self)
        return not call.said

    def recover(self, voice: Voice) -> None:
        """Say the holding line while work is open; otherwise delegate the turn with no lead-in."""
        decision = after_failure({"recovery": "talker_failed"}, work_open=self.work_open, holding=voice.lines.holding)
        if decision.action != "delegate":
            voice.turn_pending = None
        voice.actions[decision.action](self, None, decision)
        voice._rest_after(self, decision)


@dataclass
class Relay:
    """A relay: what is owed, worded into one line with the writer's rules."""

    batch: tuple[Owed, ...]
    content: str
    history: list[dict]
    note: str = ""
    version: int = 0
    # The backend results' own text when every worded item is a result: the question check reads it.
    asked: str = ""

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Return ``True``: its line is held to its end, so a result that arrives first is worded with these items."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Return whether what it words moved since it began: a fresh entry owed, its note, or the voice's version.

        A re-queued group waiting in what is owed is not a move: it is said on its own, before the batch.
        """
        fresh = any(not voice._requeued(entry) for entry in voice.owed)
        return fresh or voice.version != self.version or voice._update(self.batch)[1] != self.note

    def holds(self, delegation_id: str) -> bool:
        """Return whether it words an item of ``delegation_id``, whose end or handover then waits for its re-word."""
        return any(item.worded and item.said.delegation_id == delegation_id for item in self.batch)

    def words(self, voice: Voice, decision: TalkerDecision) -> str:
        """Return the worded line, or the worded items as written when it does not pass the checks.

        With ``keep_questions``, a line that no longer ends with the backend results' last question does not pass.
        """
        checks = {"keep_questions": voice.config.delegation.keep_questions, "language": voice.sentence_language()}
        line = relay_line(decision, self.content, self.history, asked=self.asked, **checks)
        if line is None:
            usable = relay_line(decision, self.content, self.history) is not None
            reason = "question_lost" if usable else "unusable"
            voice.observe(
                "commentary.answer_forced", talker_action=decision.action, results=len(self.batch), reason=reason
            )
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
        """Owe its items again, first (behind a re-queued group): the run the result brings words them with it."""
        voice.owe_again(self.batch, behind_requeued=True)

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Owe its items again (behind a re-queued group)."""
        voice.owe_again(self.batch, behind_requeued=True)

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


@dataclass(frozen=True)
class Pending:
    """The backend results pending when the caller's turn started: what waited, and the result lines in flight."""

    # The texts of the results owed, being worded or waiting to be said.
    waiting: tuple[str, ...]
    # The result lines not settled yet: judged by how they settle.
    playing: tuple[Line, ...]


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
        sentence_language: Callable[[], str | None] = lambda: None,
        display: DisplaySink | None = None,
        observe: Callable[..., None] = log_observation,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Wire the call's records and its delegation part; the talker is bound by :meth:`bind`.

        ``sentence_language`` reads the text-to-speech's sentence language now (its ``text_aggregation_language``), so
        the voice splits a line into the sentences the text-to-speech cuts.
        """
        self.config, self.prompts, self.lines, self.delegation = config, prompts, lines, delegation
        self.empty_turn_note, self.observe, self.clock = empty_turn_note, observe, clock
        self.sentence_language = sentence_language
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
        answers = config.delegation.answers
        self.heard_log = HeardLog(
            prompts,
            spelled_notes=config.delegation.spelled_notes,
            unheard_questions=answers.unheard_questions,
            observe=observe,
        )
        self.spoken = SpokenLines(
            self,
            reports_sentences=reports_sentences,
            clear_ended_on_cut=answers.clear_ended_on_cut,
            requeued_as_said=answers.requeued_as_said,
            observe=observe,
        )
        self.acknowledgements = Acknowledgements(self.floor, self._lead_in, observe=observe)
        self.handoffs = HandOffs(
            delegation,
            self.heard_log,
            self.worknote,
            self.display,
            self.acknowledgements,
            refused=lambda: self.notice(None, lines.refusal),
            heard=self.sent_unheard,
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
        # The last line handed to speech, any kind.
        self.last_handed: Line | None = None
        # A cut answer's unheard sentences, held for the next routing decision; the decided caller turns when it was
        # held; whether a reply was said first (then only a wait on the caller's next turn queues it).
        self._rest: Line | None = None
        self._rest_stamp = 0
        self._rest_deferred = False
        # The caller turns a routing decision decided.
        self._decided_turns = 0
        # The take is a re-queued group, said again with its words.
        self._resaying = False
        self.lead_ins: dict[str, Line] = {}
        self.instruction_due = False
        # The results pending when the caller's turn started (by identity); None once a delegation sent it or a routing
        # decision did not delegate.
        self.turn_pending: Pending | None = None
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
        self._rest = None

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
            batch = self._take_group()  # decided before any await: a re-said line never waits for a wording
            if batch is None:
                try:
                    await self._worded_early()
                except asyncio.CancelledError:
                    self._after_call()
                    raise
                batch = self._take()  # a group re-queued during the await is ruled on first
            if batch:
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
        plan.wordless, plan.steered = bool(note) and not undecided, self.instruction_due
        self._tidy()
        routing = self.layout.routing("\n\n".join(self.instructions), self.worknote.line(), turn, note)
        return TalkerCall(
            spec=self.routing_spec, request=routing, streams=plan.streams, format=self.routing_format, plan=plan
        )

    def _take(self) -> list[Owed]:
        """Take a re-queued group that may be said again with its words, alone; else everything owed, oldest first.

        Apply each end or handover whose item the batch does not word. Every item a relay takes is fresh: it is worded
        anew.
        """
        group = self._take_group()
        if group is not None:
            return group
        self._resaying = False
        batch: list[Owed] = []
        while self.owed:
            entry = self.owed.popleft()
            if isinstance(entry, Owed):
                entry.words = ""
                batch.append(entry)
            elif not any(item.worded and item.said.delegation_id == entry.delegation_id for item in batch):
                self._turnover(entry)
        return batch

    def _take_group(self) -> list[Owed] | None:
        """The first re-queued group, taken alone when it may be said again with its words; else ``None``.

        A group that may not (:meth:`_resaid`) is made fresh, so the relay words it with everything else owed.
        """
        group = self._requeued_head()
        if not group:
            return None
        reason = self._resaid(group)
        if reason:
            self.observe("answer.resay_reworded", items=len(group), delegations=self._delegations(group), reason=reason)
            self._make_fresh(group)
            return None
        for item in group:
            self.owed.remove(item)
        self._resaying = True
        return list(group)

    def _requeued_head(self) -> tuple[Owed, ...]:
        """The first re-queued group in what is owed, wherever it sits: the items in a row with the same ``words``.

        None with ``answers.requeued_as_said`` off: every owed item is fresh then.
        """
        if not self.config.delegation.answers.requeued_as_said:
            return ()
        group: list[Owed] = []
        for entry in self.owed:
            if group and not (isinstance(entry, Owed) and entry.words == group[0].words):
                break
            if isinstance(entry, Owed) and entry.words:
                group.append(entry)
        return tuple(group)

    def _requeued(self, entry: Owed | Turnover) -> bool:
        """Whether ``entry`` is an item of a re-queued group (it has its line's words), with ``requeued_as_said``."""
        return isinstance(entry, Owed) and bool(entry.words) and self.config.delegation.answers.requeued_as_said

    def _fresh_owed(self) -> tuple[Owed, ...]:
        """The items owed that a relay words: every owed item but the re-queued ones, said as they were."""
        return tuple(entry for entry in self.owed if isinstance(entry, Owed) and not self._requeued(entry))

    @staticmethod
    def _delegations(batch: Sequence[Owed]) -> list[str]:
        return list(dict.fromkeys(item.said.delegation_id for item in batch if item.said.delegation_id))

    def _resaid(self, batch: Sequence[Owed]) -> str:
        """Why ``batch``, one re-queued line's items, may not be said again with its words; ``""`` when it may.

        It may with ``answers.requeued_as_said``, no item tried ``answers.max_tries`` times, the same newer-request
        state as when it was said (``Owed.newer``), and no silent line since audio last played. The checks run in this
        order, and the first that fails is the reason: ``"tries"``, ``"newer"``, ``"silent"``.
        """
        cap = self.config.delegation.answers.max_tries
        if cap is not None and any(item.tries >= cap for item in batch):
            return "tries"
        if self.worknote.newer_state(self._delegations(batch)) != batch[0].newer:
            return "newer"
        if any(item.silent for item in batch):
            return "silent"
        return ""

    @staticmethod
    def _make_fresh(group: Sequence[Owed]) -> None:
        """``group`` may not be re-said: its words are cleared, so it is worded with what else is owed, as today."""
        for item in group:
            item.words = ""

    def _turnover(self, entry: Turnover) -> None:
        if entry.taker is None:
            self.spoken.forget(entry.delegation_id)
        else:
            self.spoken.carry_told(entry.delegation_id, entry.taker)

    def _said_by(self, batch: list[Owed]) -> TalkerCall | None:
        """Return the call that says ``batch``: a re-queued group with its words, a relay, or the lines as written."""
        if self._resaying:
            self._resaying = False
            self.observe("answer.resaid", items=len(batch), tries=max(item.tries for item in batch))
            line = self.delivered(batch, batch[0].words)
            return self.saying(line) if line else None
        if self._says_as_written(batch):
            self.observe("answer.as_written", items=len(batch))
            line = self.delivered(batch, self.joined(batch, as_written([i for i in batch if i.worded])))
            return self.saying(line) if line else None
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
        plan = Relay(tuple(batch), content, history, note, self.version, self._asked(batch))
        return TalkerCall(spec=self.answer_spec, request=relay, format=self.talker.response_format, plan=plan)

    def _update(self, batch: Sequence[Owed]) -> tuple[str, str]:
        """Return the update a relay of ``batch`` words: its items' text, and its note on what the caller heard."""
        worded = [item for item in batch if item.worded]
        delegations = list(dict.fromkeys(item.said.delegation_id for item in worded if item.said.delegation_id))
        told = " ".join(text for each in delegations if (text := self.spoken.told_text(each)))
        note = self.layout.note(told or None, self.worknote.newer_requests(delegations))
        return "\n\n".join(item.said.speak for item in worded), note

    @staticmethod
    def _asked(batch: Sequence[Owed]) -> str:
        """Return the backend results' own text when every worded item of ``batch`` is a result; else "" (not checked).

        A batch that holds commentary is never checked: commentary may be an instruction to the writer.
        """
        worded = [item for item in batch if item.worded]
        if not worded or not all(item.answers for item in worded):
            return ""
        return "\n\n".join(item.said.speak for item in worded)

    def _says_as_written(self, batch: Sequence[Owed]) -> bool:
        """Return whether ``batch`` is said as the backend wrote it, with no writer call (``result_wording``).

        Only when every worded item is a backend result and none came out silent: a silent batch goes through the
        writer first.
        """
        if self.config.delegation.result_wording != "as_written":
            return False
        worded = [item for item in batch if item.worded]
        return bool(worded) and all(item.answers for item in worded) and not any(item.silent for item in batch)

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
        batch = self._fresh_owed()
        early = self.early
        if not any(item.worded for item in batch) or self._says_as_written(batch):
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
            if early.stamp != self._stamp(self._fresh_owed()):
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
        newer = self.worknote.newer_state(list(ids))
        for item in batch:
            item.words, item.newer = text, newer
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
        self._decided_turns += bool(plan.covers)
        self.instruction_due = False

    def _hold_rest(self, line: Line) -> None:
        """A cut result line left unheard sentences: hold them, as written, for the next routing decision."""
        if not self.config.delegation.answers.unheard_rest_once or line.rest_of is not None:
            return
        language = self.sentence_language()
        rest = unheard_rest(line, language=language)
        if not alnum_only(rest):
            return
        if self._rest is not None:
            self.observe("answer.rest_dropped", line=self._rest.rest_of.number, reason="replaced")
        self._rest = Line(LineKind.RESULT, rest, line.delegation_ids, results=True, again=False, rest_of=line)
        self._rest_stamp, self._rest_deferred = self._decided_turns, False
        self.observe("answer.rest_held", line=line.number, sentences=len(sentences(rest, language=language)))

    def _rest_after(self, plan: Routing, decision: TalkerDecision) -> None:
        """The first routing decision after a cut rules on the held rest: queued, held again, or dropped.

        A ``wait`` queues it; a ``speak`` with words holds it once more, for the decision on the caller's next turn;
        any other decision drops it, as does a newer result of its delegation owed by then. No rule reads the reply's
        words.
        """
        rest = self._rest
        if rest is None:
            return
        reason = ""
        if any(isinstance(e, Owed) and e.answers and e.said.delegation_id in rest.delegation_ids for e in self.owed):
            reason = "newer_result"
        elif plan.steered:
            reason = "steered"
        elif decision.action == "wait":
            self._rest = None
            self.observe("answer.rest_said", line=rest.rest_of.number, action=decision.action)
            self.say(rest)
            return
        elif decision.action == "delegate":
            reason = "delegate"
        elif plan.wordless:
            reason = "wordless_speak"
        elif not self._rest_deferred:
            self._rest_deferred = True
            self.observe("answer.rest_deferred", line=rest.rest_of.number)
            return
        else:
            reason = "speak"
        self._rest = None
        self.observe("answer.rest_dropped", line=rest.rest_of.number, reason=reason)

    def _rest_unruled(self) -> None:
        """A decided caller turn no decision ruled on: a rest held for two decided turns is stale, and dropped."""
        if self._rest is not None and self._decided_turns - self._rest_stamp >= 2:
            self.observe("answer.rest_dropped", line=self._rest.rest_of.number, reason="stale")
            self._rest = None

    def hand(self, line: Line) -> str:
        """Hand ``line`` to speech as the response in progress, and return its words.

        The heard log notes it at once, so a delegation made while it plays carries it; playout corrects the entry.
        """
        said_as, line.number = line.number, self.responses
        self.last_handed = line
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
        """Note the caller speaking: drop the lead-ins not said yet, and say nothing meanwhile.

        The results still pending now are kept (``turn_pending``): the delegation this turn makes states them unheard
        even if they play before it is handed over. An earlier start's snapshot is kept only while a turn with words is
        still undecided; one from a start that brought no words (a cough) is taken again.
        """
        if self.turn_pending is None or not self._undecided():
            self.turn_pending = self._pending()
        self.floor.caller(True)
        self.acknowledgements.drop_all()
        self._withdraw()
        rest = next((line for line in self.waiting_lines if line.rest_of is not None), None)
        if rest is not None:  # not said yet: held again, for the decision on the caller's newer words
            self.waiting_lines.remove(rest)
            self._rest, self._rest_deferred = rest, True
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

    def unheard_sentences(self, line: Line) -> list[str]:
        """Return ``line``'s unheard rest, split into sentences in the speech's language.

        This is the one split of a line's unheard part that the voice and the heard fact use.
        """
        language = self.sentence_language()
        return sentences(unheard_rest(line, language=language), language=language)

    def heard(self, line: Line, cut: bool) -> None:
        """Settle ``line`` as heard, even in part: a cut line keeps what played, and its items are said.

        A result line's unheard sentences set the heard log's answer state (``result_settled``).
        """
        if cut:
            self.heard_log.cut(line.number, line.heard, unheard=self._unheard_questions(line))
        elif line.rest_of is not None:  # the rest was heard: the cut answer reads whole again
            self.heard_log.whole(line.rest_of.number, line.rest_of.text, rest=line.text)
        if line.results:
            self._result_settled(line, UnheardHow.CUT if cut else None)
        if cut and line.results:
            self._hold_rest(line)
        for entry in self.owed:  # audio played: a line that came out silent may be said again as it was
            if isinstance(entry, Owed):
                entry.silent = False
        self._done(line.items)
        self._moved()

    def not_heard(self, line: Line) -> None:
        """Settle ``line`` as not heard: it leaves the heard log, and its items are owed again, first.

        Each item counts a try; a backend answer tried ``answers.max_tries`` times is dropped and the backend is told
        (:meth:`_drop_unheard`). A line said once more after it produced no audio marks its items silent. When some
        items of the line are dropped, the others are made fresh: their words still hold the dropped answer's text.
        """
        self.heard_log.unsay(line.number)
        if line.items:
            cap = self.config.delegation.answers.max_tries
            again: list[Owed] = []
            for item in line.items:
                item.tries += 1
                item.silent = item.silent or line.retried
                if item.answers and cap is not None and item.tries >= cap:
                    self._drop_unheard(item)
                else:
                    again.append(item)
            if again:
                if len(again) < len(line.items):
                    self._make_fresh(again)
                self.owe_again(again)
                tries = max(item.tries for item in again)
                self.observe("answer.requeued", line=line.number, items=len(again), tries=tries)
                self.request_run()
        self._move()

    def _drop_unheard(self, item: Owed) -> None:
        """``item``, a backend answer, went unheard ``answers.max_tries`` times: not said again; the backend is told.

        The work note counts it as said, so it no longer reads "answered"; the heard log's answer state reads it as
        dropped, and the next delegation reads ``transcript.not_said`` with the backend's text. Its developer message
        stays out of the context.
        """
        self.observe("answer.dropped_unheard", delegation_id=item.said.delegation_id, tries=item.tries)
        self._done([item])
        self.heard_log.result_unsaid(item.said.speak)
        self.heard_log.not_said(item.said.speak)
        if item.worded:
            self.unsaid.add(id(item.message))
            self._take_out({id(item.message)})

    def _unheard_questions(self, line: Line) -> tuple[str, ...]:
        """The questions of a result line that did not play: the sentences of its unheard rest that ask."""
        if not line.results:
            return ()
        questions = tuple(sentence for sentence in self.unheard_sentences(line) if ends_with_question(sentence))
        if questions:
            self.observe("answer.question_unheard", line=line.number, questions=len(questions))
        return questions

    def _result_settled(self, line: Line, how: UnheardHow | None) -> None:
        """A result line settled: what of it the caller did not hear sets the heard log's answer state."""
        replaces = line.rest_of.number if line.rest_of is not None else None
        settled = self.heard_log.result_settled(line.number, self.unheard_sentences(line), how, replaces=replaces)
        if settled.reply is not ReplyUnheard.NONE:
            how = settled.how.value if settled.how is not None else ""
            self.observe("answer.unheard_part", line=line.number, reply_unheard=settled.reply.value, unheard_how=how)

    def _level(self, text: str) -> ReplyUnheard:
        """The level of a whole text the caller has not heard."""
        return ReplyUnheard.of(sentences(text, language=self.sentence_language()))

    def _unheard_results(self) -> list[str]:
        """The texts of results the caller has not heard in full: owed, being worded, waiting, or not settled."""
        texts = [entry.said.speak for entry in self.owed if isinstance(entry, Owed) and entry.answers]
        plan = self.call.plan if self.call is not None else None
        if isinstance(plan, Relay):
            texts += [item.said.speak for item in plan.batch if item.answers]
        lines = [*self.waiting_lines, *self.spoken.unsettled()]
        if isinstance(plan, Saying):
            lines.append(plan.line)
        return texts + [line.text for line in lines if line.results]

    def reply_unheard(self) -> Unheard:
        """What of the backend's answers the caller has not heard now.

        The worse of the results still waiting (owed, being worded, waiting to be said, or not settled) and every
        result line that settled since the last delegation (``HeardLog.since_sent``).
        """
        unheard = self.heard_log.since_sent
        for text in self._unheard_results():
            unheard = unheard.worse(Unheard(self._level(text), UnheardHow.WAITING))
        return unheard if unheard.reply is not ReplyUnheard.NONE else HEARD

    def _pending(self) -> Pending:
        """The results pending now, by identity."""
        waiting = [entry.said.speak for entry in self.owed if isinstance(entry, Owed) and entry.answers]
        plan = self.call.plan if self.call is not None else None
        if isinstance(plan, Relay):
            waiting += [item.said.speak for item in plan.batch if item.answers]
        lines = [*self.waiting_lines, *([plan.line] if isinstance(plan, Saying) else [])]
        waiting += [line.text for line in lines if line.results]
        playing = tuple(line for line in self.spoken.unsettled() if line.results)
        return Pending(tuple(waiting), playing)

    def sent_unheard(self) -> Unheard:
        """What a delegation handed over now states: the worse of the turn-start snapshot, judged now, and the reading.

        A result that waited when the caller's turn started counts whole, even if it played since: the caller spoke
        before hearing it. A line in flight then is judged by how it has settled since (its unheard sentences), and
        counts whole while it is still unsettled. The snapshot is cleared: the next turn takes its own.
        """
        now, pending, self.turn_pending = self.reply_unheard(), self.turn_pending, None
        if pending is None:
            return now
        snapshot = HEARD
        for text in pending.waiting:
            snapshot = snapshot.worse(Unheard(self._level(text), UnheardHow.WAITING))
        for line in pending.playing:
            if line.outcome in (Outcome.HEARD, Outcome.CUT, Outcome.NOT_HEARD):
                how = UnheardHow.WAITING if line.outcome is Outcome.NOT_HEARD else UnheardHow.CUT
                level = ReplyUnheard.of(self.unheard_sentences(line))
            else:
                how, level = UnheardHow.WAITING, self._level(line.text)
            snapshot = snapshot.worse(Unheard(level, how))
        if snapshot.reply is ReplyUnheard.NONE or now.reply.at_least(snapshot.reply):
            return now
        return Unheard(snapshot.reply, snapshot.how, from_turn_start=True)

    def say_again(self, line: Line) -> None:
        """Say ``line`` once more, behind the lines already waiting: it produced no audio."""
        self.say(line)

    def owe_again(self, items: Sequence[Owed], *, behind_requeued: bool = False) -> None:
        """Owe ``items`` again, first; until a line says them, their messages stay out of the context.

        ``behind_requeued``: right after the re-queued group at the head of what is owed, which is ruled on first.
        """
        group = self._requeued_head() if behind_requeued else ()
        at = len(group) if group and self.owed and self.owed[0] is group[0] else 0
        for offset, item in enumerate(items):
            self.owed.insert(at + offset, item)
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

    def answer_due(self) -> bool:
        """Whether an answer is about to be heard.

        A result, or commentary that answers a client delegation, owed or being worded; a result line waiting, being
        said or in flight; or a held rest.
        """
        plan = self.call.plan if self.call is not None else None
        batch = plan.batch if isinstance(plan, Relay) else ()
        owed = any(
            isinstance(entry, Owed)
            and (entry.answers or (entry.kind is LineKind.COMMENTARY and entry.said.delegation_id))
            for entry in (*self.owed, *batch)
        )
        lines = [*self.waiting_lines, *self.spoken.lines.values()]
        if isinstance(plan, Saying):
            lines.append(plan.line)
        return owed or any(line.kind is LineKind.RESULT for line in lines) or self._rest is not None

    def _skip_lead_in(self, text: str) -> str:
        """Why a lead-in is not said, in this order; "" when it is (``answers.lead_in_skip``).

        ``answer_due``: an answer is about to be heard; ``lead_in_waiting``: another lead-in waits or plays;
        ``same_as_last``: it repeats the last line handed, by letters and digits. A lead-in after one that played is
        said: it acknowledges another request.
        """
        if not self.config.delegation.answers.lead_in_skip:
            return ""
        if self.answer_due():
            return "answer_due"
        if any(line.kind is LineKind.LEAD_IN for line in [*self.waiting_lines, *self.spoken.lines.values()]):
            return "lead_in_waiting"
        last = self.last_handed
        if last is not None and alnum_only(text) == alnum_only(last.text):
            return "same_as_last"
        return ""

    def _lead_in(self, delegation_id: str, text: str) -> None:
        """Say a delegation's lead-in, which tells the caller about it, unless it is skipped (:meth:`_skip_lead_in`)."""
        reason = self._skip_lead_in(text)
        if reason:
            self.observe("acknowledgement.skipped", delegation_id=delegation_id, reason=reason)
            return
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
                self.heard_log.partly(line.number, line.heard, unheard=self._unheard_questions(line))
                if line.results:
                    self._result_settled(line, UnheardHow.NO_AUDIO)
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
