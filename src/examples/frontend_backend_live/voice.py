# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One call's voice: which run of the talker each one is, what its decision does, and what the backend sends.

The talker is Pipecat's LLM service in the pipeline (``examples.frontend_backend_live.talker``), its memory the context
the aggregators keep, bounded by its token budget (:meth:`Voice.bound`). :class:`Voice` is what the talker asks before
and after each run (``talker.Calls``):

* **Which call.** Caller words not decided yet, or an instruction, make a routing call; a line waiting to be said (a
  lead-in, the holding line, a line said once more) is said with no model call; what is owed (results, commentary,
  fixed lines) makes one relay; a progress line comes last. A run with nothing to do sends no request. A turn with no
  words right after a cut (Pipecat's empty-turn run, ``speech.resume_after_noise``) says the cut line's unheard rest.
* **What the decision does.** Each call carries its plan (:class:`Routing`, :class:`Relay`, :class:`Progress`,
  :class:`Saying`), which acts on the decision, a cancellation or a failure. A routing ``speak`` has streamed;
  ``delegate`` hands the turn over with its speech as the lead-in (``examples.frontend_backend_live.handoffs``);
  ``wait`` says nothing, but resumes a line the turn's words cut. Output that cannot be used, or a call that fails (from
  the pipeline's error event), gives the holding line while work is open, else a delegation with no lead-in.
* **Speech when idle.** The voice listens on the session's delegation part (``gateways.live.ports.DelegationPort``)
  through the delegation reader (``examples.frontend_backend_live.delegations``). Each result and each piece of
  commentary is owed and runs the talker as Pipecat's own backend connector delivers
  (``pipecat.pipeline.llm_with_backend.connector``): an empty append, uninterruptible, queued at the talker with
  ``run_llm``, which the assistant aggregator holds while the bot speaks, owes while the caller speaks, and bundles
  with what queues behind it. The run's relay words everything owed in one line, and the items' developer messages go
  into the talker's context with that line, right before it (:meth:`Voice.written`). Thinking and typed questions are
  written with no run. A fixed line (an ending's, the refusal, the app-wait line) and a progress line wait the same
  way.
* **Worded early.** With the ``writer`` wording, what is owed is worded as soon as it is queued, even while the caller
  speaks, a turn is decided or a line plays: one relay of the talker, run out of band
  (``examples.frontend_backend_live.talker.DecisionReplyMixin.run_call``). The relay at idle says that line with no
  request while its stamp holds: the same items, the same notes, and nothing else moved since. A caller turn alone
  keeps the stamp; a line, an instruction, thinking, a cut, a new request or a new item moves it, and a move cuts a
  wording under way short. Everything owed is worded again once no caller turn waits or is decided, so after a reply
  or a lead-in, while that line plays.

The voice keeps the lines it said (``examples.frontend_backend_live.spoken``) and the records the delegations read: the
work note and the heard log.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from loguru import logger
from pipecat.frames.frames import InterruptionFrame, LLMMessagesAppendFrame
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_live.delegations import DelegationReader, ReadyResult
from examples.frontend_backend_live.display import NullDisplay
from examples.frontend_backend_live.handoffs import Acknowledgements, HandOffs
from examples.frontend_backend_live.heard_log import HeardLog
from examples.frontend_backend_live.models.decision import ACTIONS, CallSpec, TalkerDecision, decision_from_output_text
from examples.frontend_backend_live.normalizer import TurnNormalizer
from examples.frontend_backend_live.spoken import CutMemory, Line, LineKind, SpokenLines
from examples.frontend_backend_live.talker import (
    Layout,
    TalkerCall,
    after_failure,
    floored,
    plain_speech,
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
    from examples.frontend_backend_live.talker import TalkerChat
    from gateways.live.ports import DelegationPort


def message_text(message: dict) -> str:
    """A context message's text, its parts joined when the content is a list."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def message_tokens(message: dict) -> int:
    """A cheap token estimate for one message, its role included: :func:`approx_tokens` of its JSON."""
    return approx_tokens(json.dumps(message, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------- the context
@dataclass(frozen=True)
class Kind:
    """What a message of the talker's context is to the talker, and how its calls read it."""

    name: str
    role: str  # the role the talker's calls read it in
    reference: bool = False  # reference data: a caller turn right after it carries the caller label
    turn: bool = False  # a caller turn, which a routing call decides


CALLER = Kind("caller", "user", turn=True)  # the user aggregator's
SPOKEN = Kind("spoken", "assistant")  # what a line played: the assistant aggregator's
RESULT = Kind("result", "user", reference=True)  # a backend result
APP_CONTEXT = Kind("app_context", "user", reference=True)  # commentary, thinking, the greeting's instructions
TYPED = Kind("typed", "user", reference=True)  # a question the application typed itself
# The messages the aggregators write, by role; any other the voice did not write is a note no call reads.
WRITTEN = {"user": CALLER, "assistant": SPOKEN}


# ---------------------------------------------------------------------------------------------- what is owed
@dataclass(frozen=True)
class Instruction(ReadyResult):
    """Instructions worded into a line, such as the greeting's; when they cannot be, ``fallback`` is said instead."""

    fallback: str  # the line said in their place: the instructions themselves are never said

    @property
    def unworded(self) -> str:
        """The fallback line."""
        return self.fallback


@dataclass(eq=False)
class Owed:
    """Something to say: a worded item (its developer message goes into the context with its line) or a fixed line."""

    said: ReadyResult
    kind: LineKind  # the line kind it gives: RESULT, COMMENTARY or NOTICE
    message: dict | None = None  # the developer message of a worded item
    answers: bool = False  # a delegation's result: answered in the work note until it is heard

    @property
    def worded(self) -> bool:
        """Worded by the relay, or said as written."""
        return self.message is not None


@dataclass(frozen=True)
class Turnover:
    """The end of a delegation, or its handover to ``taker``, applied in its turn."""

    delegation_id: str
    taker: str | None = None


# A mixed line takes the first of these kinds it carries.
LINE_KIND_ORDER = (LineKind.RESULT, LineKind.COMMENTARY, LineKind.NOTICE)


def as_written(items: Sequence[Owed]) -> str:
    """The items as the backend or the application wrote them, as plain spoken sentences, in order."""
    return " ".join(line for line in (plain_speech(item.said.unworded) for item in items) if line)


# ---------------------------------------------------------------------------------------------- the plans
# Each call of the talker carries its plan, which meets its decision, a cut by a result, a cancellation and a failure.
@dataclass
class Routing:
    """A routing call: a decision on the caller's undecided turns, or an instruction."""

    covers: tuple[dict, ...]  # the caller turns it decides
    request: str  # the words a delegation carries as its request
    work_open: bool  # whether work was open when it began: a failure holds instead of delegating
    cut: Line | None  # the line the turns' words cut, which a wait resumes

    streams: ClassVar[bool] = True

    def cut_by_result(self) -> bool:
        """Only a decision on caller turns reads the work note a result would change."""
        return bool(self.covers)

    def moved(self, voice: Voice) -> bool:
        """Never: only a result cuts a decision."""
        return False

    def holds(self, delegation_id: str) -> bool:
        """Never: it words no item."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Carry the decision out; the hand-off starts with nothing awaited before it."""
        if call.said:  # its words already went to speech: the turn is answered, whatever followed
            decision = TalkerDecision("speak", call.said, decision.diagnostics)
        decision = floored(decision, work_open=self.work_open, holding=voice.lines.holding)
        voice.decided_by(self)
        voice.observe("talker.completed", action=decision.action, **decision.diagnostics)
        return voice.actions[decision.action](self, call, decision)

    def dropped(self, voice: Voice) -> None:
        """Its turns wait for the run the result brings, which decides them with the result in view."""
        voice.observe("talker.cut_off")

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Its turns stay undecided, unless its words went to speech."""
        if call.said:
            voice.decided_by(self)

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Its turns are taken now; the fallback is due unless its words went to speech."""
        voice.decided_by(self)
        return not call.said

    def recover(self, voice: Voice) -> None:
        """The holding line while work is open; otherwise the turn is delegated with no lead-in."""
        decision = after_failure({"recovery": "talker_failed"}, work_open=self.work_open, holding=voice.lines.holding)
        voice.actions[decision.action](self, None, decision)


@dataclass
class Relay:
    """A relay: what is owed, worded into one line with the writer's rules."""

    batch: tuple[Owed, ...]
    content: str  # the worded items' text, which the update holds
    history: list[dict]  # the conversation the relay read
    note: str = ""  # its update's note on what the caller already heard
    version: int = 0  # the voice's version when it began

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Always: its line is held to its end, so a result that arrives before it is worded with these items."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Whether what it words moved since it began: a new entry owed, its note, or the voice's version.

        A move cuts a wording under way short, so everything owed is worded again at once.
        """
        return bool(voice.owed) or voice.version != self.version or voice._update(self.batch)[1] != self.note

    def holds(self, delegation_id: str) -> bool:
        """Whether it words an item of ``delegation_id``: that delegation's end or handover waits, for its re-word."""
        return any(item.worded and item.said.delegation_id == delegation_id for item in self.batch)

    def words(self, voice: Voice, decision: TalkerDecision) -> str:
        """The worded line, or the worded items as written when it does not pass the checks."""
        line = relay_line(decision, self.content, self.history)
        if line is None:
            voice.observe("commentary.answer_forced", talker_action=decision.action, results=len(self.batch))
            line = as_written([item for item in self.batch if item.worded])
        return line

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """The worded line, or the items as written when it does not pass the checks; nothing once it moved."""
        if self.moved(voice):
            self.dropped(voice)
            return ""
        delivered = voice.delivered(self.batch, voice.joined(self.batch, self.words(voice, decision)))
        return voice.hand(delivered) if delivered else ""

    def dropped(self, voice: Voice) -> None:
        """Its items are owed again, first: the run the result brings words them with it, in one line."""
        voice.owe_again(self.batch)

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Its items are owed again."""
        voice.owe_again(self.batch)

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """The items are said as written."""
        return True

    def recover(self, voice: Voice) -> None:
        """The items as written, said with no model call."""
        line = voice.delivered(self.batch, voice.joined(self.batch, as_written([i for i in self.batch if i.worded])))
        if line is not None:
            voice.say(line)


@dataclass
class Progress:
    """A progress line about one delegation's quiet work."""

    entry: OpenWork
    version: int  # the voice's version when it was offered: said only if nothing moved since

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """A result moves what the line is about."""
        return True

    def moved(self, voice: Voice) -> bool:
        """Whether anything moved since it was offered: a new item, or the voice's version."""
        return voice.version != self.version or any(isinstance(entry, Owed) for entry in voice.owed)

    def holds(self, delegation_id: str) -> bool:
        """Never: it words no item."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """Say the line only if nothing moved while it was worded; its news is taken either way then."""
        delegation_id = self.entry.id
        if self.moved(voice) or not voice.worknote.quiet(delegation_id):
            voice.observe("progress.dropped", delegation_id=delegation_id)  # its news stays for the next quiet spell
            return ""
        voice.worknote.worded(delegation_id)
        text = progress_line(decision, self.entry.progress)
        if not text:
            voice.observe("progress.silent", delegation_id=delegation_id)
            return ""
        voice.observe("progress.said", delegation_id=delegation_id)
        return voice.hand(Line(LineKind.STATUS, text, (delegation_id,), tells=(delegation_id,), again=False))

    def dropped(self, voice: Voice) -> None:
        """Its news stays for the next quiet spell."""
        voice.observe("progress.dropped", delegation_id=self.entry.id)

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Nothing: a progress line holds no fact to keep."""

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """Nothing is said."""
        return False

    def recover(self, voice: Voice) -> None:
        """Nothing."""


@dataclass
class Saying:
    """A line said with no model call: a lead-in, a fixed line, a resumed rest, a line said once more."""

    line: Line

    streams: ClassVar[bool] = False

    def cut_by_result(self) -> bool:
        """Never: it asks no model."""
        return False

    def moved(self, voice: Voice) -> bool:
        """Never: it asks no model."""
        return False

    def holds(self, delegation_id: str) -> bool:
        """Never: what it says is delivered already."""
        return False

    def finished(self, voice: Voice, call: TalkerCall, decision: TalkerDecision | None) -> str:
        """The line goes to speech."""
        return voice.hand(self.line)

    def dropped(self, voice: Voice) -> None:
        """Not cut."""

    def cancelled(self, voice: Voice, call: TalkerCall) -> None:
        """Nothing: the cut that cancelled it settles it as not heard."""

    def failed(self, voice: Voice, call: TalkerCall) -> bool:
        """It cannot fail: it sends no request."""
        return False

    def recover(self, voice: Voice) -> None:
        """Nothing."""


# ---------------------------------------------------------------------------------------------- the result wordings
class ResultWording(Protocol):
    """How what is owed becomes one line, by the name ``delegation.result_wording`` gives."""

    offers_progress: ClassVar[bool]  # whether a quiet spell may bring a progress line
    words_early: ClassVar[bool]  # whether what is owed is worded as soon as it is queued

    def call(self, voice: Voice, batch: list[Owed]) -> TalkerCall | None:
        """The call that says ``batch``, which holds at least one worded item."""


class Writer:
    """``writer``: the talker's relay words every worded item in one line, with the writer's rules."""

    offers_progress: ClassVar[bool] = True
    words_early: ClassVar[bool] = True

    def call(self, voice: Voice, batch: list[Owed]) -> TalkerCall | None:
        """The line worded early while its stamp holds, else the relay: the conversation, then the update."""
        return voice.relay(batch)


class AsWritten:
    """``as_written``: every worded item said as written, cleaned for speech; no progress line."""

    offers_progress: ClassVar[bool] = False
    words_early: ClassVar[bool] = False

    def call(self, voice: Voice, batch: list[Owed]) -> TalkerCall | None:
        """The items as written, said with no model call."""
        line = voice.delivered(batch, voice.joined(batch, as_written([item for item in batch if item.worded])))
        return voice.saying(line) if line else None


RESULT_WORDINGS: dict[str, type[ResultWording]] = {"writer": Writer, "as_written": AsWritten}


# ---------------------------------------------------------------------------------------------- the voice
class Floor:
    """When the caller speaks, and how long the voice has been idle."""

    def __init__(self, clock: Callable[[], float]):
        """Start idle, with the caller silent."""
        self.clock = clock
        self.caller_speaking = False
        self.caller_starts = 0  # how many times the caller started speaking
        self.idle = True
        self.idle_since = clock()
        self._changed = asyncio.Event()

    def caller(self, speaking: bool) -> None:
        """The caller started or stopped speaking."""
        self.caller_speaking = speaking
        self.caller_starts += speaking
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    def settle(self, idle: bool) -> None:
        """Whether the voice is idle now: the caller silent, no call running, nothing in flight or due."""
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


class OneShot:
    """The talker's one-shot calls as a decision maker, for the normalizer: the same layout and format, no stream."""

    def __init__(self, talker: TalkerChat, layout: Layout):
        """Call ``talker`` out of band, with ``layout``'s messages."""
        self.talker, self.layout = talker, layout

    async def complete(self, instructions: str, history: list[dict], spec: CallSpec) -> TalkerDecision:
        """One decision for ``history`` under ``instructions``: a call bound to Pipecat's ``run_inference``."""
        messages = self.layout.chat_messages(instructions, history)
        call = TalkerCall(spec=spec, messages=messages, format=self.talker.response_format)
        return decision_from_output_text(await self.talker.run_call(call) or "", {})


class Voice:
    """One call's voice: chooses each run of the talker and acts on its decision; says what the backend sends."""

    def __init__(
        self,
        *,
        config: LiveConfig,
        prompts: PromptSet,
        lines: FixedLines,
        json_rules: str,
        resume_note: str,
        delegation: DelegationPort,
        wording: ResultWording,
        cut_memory: CutMemory,
        item_text: Callable[[dict], str],
        reports_sentences: bool = True,
        display: DisplaySink | None = None,
        observe: Callable[..., None] = log_observation,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Wire the call's records and its delegation part; the talker is bound by :meth:`bind`.

        Args:
            config: The validated ``frontend.cascade`` block of ``config.yaml``.
            prompts: The prompt set of the session's read of the example's prompt file.
            lines: The voice's fixed lines.
            json_rules: The decision rules, after every call's instructions.
            resume_note: The note Pipecat writes for a turn with no words that cut a line (``EmptyUserTurnConfig``).
            delegation: The session's delegation part.
            wording: How what is owed becomes one line (``delegation.result_wording``).
            cut_memory: What the talker's context keeps of a line cut while it played (``speech.heard_words``).
            item_text: A message item's text, which the delegation reader reads results with.
            reports_sentences: Whether the text-to-speech's mode reports each sentence after its audio
                (``examples.frontend_backend_live.media.services.SpeechMode``).
            display: Where the display part of each delegation goes; nothing is shown by default.
            observe: ``observe(kind, **fields)`` hook for traces.
            clock: The call's one clock.
        """
        self.config, self.prompts, self.lines, self.delegation = config, prompts, lines, delegation
        self.resume_note, self.wording, self.observe, self.clock = resume_note, wording, observe, clock
        self.layout = Layout(prompts, json_rules)
        self.display = display or NullDisplay()
        speech = config.speech
        self.routing_spec = CallSpec("talker", ACTIONS, prompts.call_hints.talker)
        self.answer_spec = CallSpec("writer", ("speak",), prompts.call_hints.answer)
        self.progress_spec = CallSpec("writer", ("speak",), prompts.call_hints.progress)
        # What each of the routing decision's actions does with the turns it decided.
        self.actions: dict[str, Callable[[Routing, TalkerCall | None, TalkerDecision], str]] = {
            "speak": self._spoke,
            "delegate": self._delegated,
            "wait": self._waited,
        }
        self.floor = Floor(clock)
        self.worknote = WorkNote(prompts, tracks=delegation.reports_events, clock=clock, observe=observe)
        self.heard_log = HeardLog(prompts)
        self.spoken = SpokenLines(
            self,
            cut_memory=cut_memory,
            start_timeout_seconds=speech.start_timeout_seconds,
            retry_silent_lines=speech.retry_silent_lines,
            reports_sentences=reports_sentences,
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
            observe=observe,
        )
        self.reader = DelegationReader(
            self.worknote,
            self,
            self.display,
            lines,
            display_label=config.display_label,
            app_wait_seconds=config.delegation.app_wait_seconds,
            item_text=item_text,
            observe=observe,
            outcome_arrived=self.acknowledgements.drop,
        )
        self.worknote.on_answer = self.result_arrived
        self.talker: TalkerChat | None = None
        self.context: LLMContext | None = None
        self.normalizer: TurnNormalizer | None = None
        self.instructions: list[str] = []  # the seed's system and developer texts, then each instruction append
        self.seed_roles: list[str] = []  # the role of each of the seed's texts among them
        self.kinds: dict[int, tuple[dict, Kind]] = {}  # the context's messages the voice wrote, by identity
        self.decided: dict[int, dict] = {}  # the caller turns a routing call decided, by identity
        self.noted: dict[int, Line | None] = {}  # the caller turns noted in the heard log: the line each one cut
        self.unsaid: set[int] = set()  # the worded items' messages not said yet: no call reads them
        self.sent: list[dict] = []  # the voice's appends on their way to the context, which the calls read already
        self._writes: list[dict] = []  # the items' messages that go into the context with the line just handed
        self.owed: deque[Owed | Turnover] = deque()
        self.early: EarlyLine | None = None  # what is owed, worded early
        self.waiting_lines: deque[Line] = deque()  # lines said with no model call, in turn
        self.lead_ins: dict[str, Line] = {}  # the lead-ins among them, by delegation: the caller's words take them back
        self.instruction_due = False  # an instruction append waits for its routing call
        self.progress_due: OpenWork | None = None
        self.call: TalkerCall | None = None
        self.failures: deque[TalkerCall] = deque()  # calls whose fallback runs from the pipeline's error event
        self.responses = 0  # the talker's responses so far: each one's number
        self.run_requested = False
        self.version = 0  # moves with what every wording reads but caller turns: lines, instructions, thinking
        self._on_purpose = False  # the next interruption is an instruction's
        self._handled_notes: dict[int, dict] = {}  # Pipecat's empty-turn notes already acted on, by identity
        self._offered_at = float("-inf")  # when the quiet task last offered a progress line
        self._made = 0
        self._tasks: set[asyncio.Task] = set()
        self._quiet_task: asyncio.Task | None = None
        self._closed = False
        delegation.listen(self.reader)

    # ------------------------------------------------------------------------------ the build's side
    def seed(self, messages: Sequence[dict]) -> list[dict]:
        """Take the host's seed: its system and developer texts are the instructions; the rest starts the context.

        The bound never drops the instructions, as the Live port requires (``gateways.live.ports.FrontendAdapter``). A
        user message under one of the set's labels is that kind of reference data again; the seed's caller turns are
        history.
        """
        conversation = []
        labels = {self.prompts.result_label: RESULT, self.prompts.context_label: APP_CONTEXT}
        for message in messages:
            text = message_text(message)
            if message.get("role") in ("system", "developer"):
                self.instructions.append(text)
                self.seed_roles.append(message["role"])
                continue
            conversation.append(message)
            kind = next((kind for label, kind in labels.items() if text.startswith(label)), None)
            if message.get("role") == "user" and kind is not None:
                self._register(message, kind)
            self.decided[id(message)] = message
        return conversation

    def show(self, display: DisplaySink) -> None:
        """Show each delegation on ``display``, a sink made once the pipeline's worker is built."""
        self.display = self.reader.display = self.handoffs.display = display

    def bind(self, talker: TalkerChat, context: LLMContext) -> None:
        """Run ``talker`` on ``context``: the talker asks this voice before and after each run."""
        self.talker, self.context = talker, context
        talker.calls = self
        if self.config.normalizer.enabled:
            self.normalizer = TurnNormalizer(OneShot(talker, self.layout), self.prompts, self.config.normalizer)

    def start(self) -> None:
        """Start the quiet task that offers progress lines. Once only; the host that built the voice starts it."""
        progress_after = self.config.delegation.progress_after_seconds
        if self._quiet_task is None and progress_after is not None and self.wording.offers_progress:
            self._quiet_task = asyncio.create_task(self._quiet(progress_after), name="delegation-progress")

    async def close(self) -> None:
        """Ignore later delegation events, drop what waits to be said, and stop every task. Safe twice."""
        self._closed = True
        self.reader.close()
        self.acknowledgements.drop_all()
        self.handoffs.close()
        self.spoken.close()
        if self.early is not None:
            self.early.cut()
        tasks = [*self._tasks, *([self._quiet_task] if self._quiet_task else [])]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.owed.clear()
        self.waiting_lines.clear()

    # ------------------------------------------------------------------------------ the context
    def _register(self, message: dict, kind: Kind) -> None:
        self.kinds[id(message)] = (message, kind)

    def kind_of(self, message: dict) -> Kind | None:
        """What a context message is: one the voice wrote, a caller turn, a line that played, or none (a note)."""
        registered = self.kinds.get(id(message))
        if registered is not None and registered[0] is message:
            return registered[1]
        return WRITTEN.get(message.get("role"))

    def conversation(self, *, labeled: bool = False) -> list[dict]:
        """The context as the talker's calls read it: each message's role, text and kind, items not said yet left out.

        The context is bounded first (:meth:`bound`). The voice's own appends count from the moment it queues them, as
        they reach the context only behind the audio queued before them.

        ``labeled``: the routing call's view, with the caller label in front of a caller turn right after reference
        data, so a caller's "yes" right after a result reads as the caller's. Each message's kind decides, never its
        text.
        """
        self.bound()
        history, after_reference = [], False
        messages = list(self.context.get_messages()) if self.context else []
        held = {id(message) for message in messages}
        self.sent = [message for message in self.sent if id(message) not in held]  # in the context now
        for message in [*messages, *self.sent]:
            kind = self.kind_of(message)
            if kind is None or id(message) in self.unsaid:
                continue
            content = message_text(message)
            if labeled and after_reference and kind.turn:
                content = self.prompts.caller_label + content
            after_reference = kind.reference
            history.append({"role": kind.role, "content": content, "kind": kind.name})
        return history

    def bound(self) -> None:
        """Above ``memory.compact_at`` of ``memory.context_tokens``, drop the oldest turn groups from the context.

        The count is :func:`message_tokens` of each message as the talker's calls read it, the instructions included,
        which stay outside the context, and each line in flight whole, from the moment it is handed (:meth:`hand`) until
        it settles. Each caller turn opens a group, and the newest group never leaves. An item not said yet stays where
        it is, uncounted. No model is called.
        """
        if self.context is None:
            return
        memory = self.config.memory
        limit = memory.compact_at * memory.context_tokens
        messages = self.context.get_messages()
        held = {id(message) for message in messages}
        pinned = [
            {"role": self.seed_roles[i] if i < len(self.seed_roles) else "developer", "content": text}
            for i, text in enumerate(self.instructions)
        ]
        tokens = sum(map(message_tokens, pinned)) + sum(self._tokens(m) for m in self.sent if id(m) not in held)
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
            for records in (self.kinds, self.decided, self.noted, self._handled_notes):
                records.pop(key, None)
        self.context.set_messages([message for index, message in enumerate(messages) if index not in dropped])

    def _tokens(self, message: dict) -> int:
        """The estimated tokens of ``message`` as the talker's calls read it; none for a note or an item not said."""
        kind = self.kind_of(message)
        if kind is None or id(message) in self.unsaid:
            return 0
        return message_tokens({"role": kind.role, "content": message_text(message)})

    def runtime_data(self) -> str:
        """The routing call's runtime data now, with no turn named."""
        return self.layout.runtime_data(self.worknote.line())

    def _undecided(self) -> list[dict]:
        """The caller turns no routing call decided yet, oldest first."""
        messages = self.context.get_messages() if self.context else ()
        return [m for m in messages if (kind := self.kind_of(m)) and kind.turn and id(m) not in self.decided]

    def _append(self, message: dict, kind: Kind, *, run: bool) -> None:
        """Write ``message`` into the talker's context in its turn, through the pipeline; ``run``: run the talker then.

        A run already asked for covers it: the assistant aggregator bundles the appends queued behind one that runs.
        """
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
        """Queue ``frame`` at the talker, past any interruption on the way, as Pipecat's backend connector does."""
        if self.talker is None or self._closed:
            return
        frame.interruptible = False
        task = asyncio.get_running_loop().create_task(self.talker.queue_frame(frame))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ------------------------------------------------------------------------------ the talker's calls (talker.Calls)
    async def next_call(self, context: LLMContext) -> TalkerCall | None:
        """The call this run of the talker makes, by the rules in the module docstring; ``None``: no request."""
        self.run_requested = False
        call = None
        resumed = self._resumed(context)
        undecided = self._undecided()
        if resumed is not None:
            self.observe("line.resumed")
            call = self.saying(resumed)
        elif undecided or self.instruction_due:
            call = await self._routing(undecided)
        elif self.waiting_lines:
            call = self.saying(self._next_line())
        elif any(isinstance(entry, Owed) for entry in self.owed):
            try:
                await self._worded_early()
            except asyncio.CancelledError:
                self._after_call()  # an interruption: what is owed gets a run once nobody speaks
                raise
            if batch := self._take():
                call = self._said_by(batch)
        elif self.progress_due is not None:
            call = self._progress()
        self.call = call
        if call is not None and call.spec is not None:
            call.format = self.talker.response_format
            call.deadline = asyncio.get_running_loop().time() + self.config.reliability.talker_attempt_timeout_seconds
        self._moved()
        return call

    def _resumed(self, context: LLMContext) -> Line | None:
        """After Pipecat's note for a turn with no words that cut a line, that line's unheard rest, once."""
        messages = context.get_messages()
        note = messages[-1] if messages else None
        if note is None or self._handled_notes.get(id(note)) is note or self.kind_of(note) is not None:
            return None
        if note.get("content") != self.resume_note:
            return None
        self._handled_notes[id(note)] = note
        return self.spoken.rest()

    async def _routing(self, undecided: list[dict]) -> TalkerCall:
        """The routing call on the caller's undecided turns (and an instruction): heard, normalized, then laid out."""
        for message in undecided:
            if id(message) not in self.noted:
                self.heard_log.caller(message_text(message))
                self.noted[id(message)] = self.spoken.caller_spoke()
        if undecided and self.normalizer is not None:
            await self._normalize(undecided[-1])
        turn = " ".join(message_text(message) for message in undecided)
        request = turn or (self.instructions[-1] if self.instructions else "")
        cut = next((self.noted[id(m)] for m in undecided if self.noted.get(id(m)) is not None), None)
        plan = Routing(tuple(undecided), request, bool(self.worknote.open()), cut)
        messages = self.layout.routing(
            "\n\n".join(self.instructions), self.conversation(labeled=True), self.worknote.line(), turn
        )
        return TalkerCall(spec=self.routing_spec, messages=messages, streams=plan.streams, plan=plan)

    async def _normalize(self, message: dict) -> None:
        """Rewrite the newest caller turn in written form, where its prompt finds details to rewrite."""
        heard = message_text(message)
        normalized, diagnostics = await self.normalizer.normalize(self.conversation(), heard)
        self.observe("normalizer.completed", heard=heard, normalized=normalized, **diagnostics)
        if normalized != heard:
            message["content"] = normalized
            self.heard_log.replace_caller(heard, normalized)

    def _take(self) -> list[Owed]:
        """Everything owed, oldest first; ends and handovers are applied in their turn.

        An end or a handover of a delegation whose worded item the batch already holds needs nothing: saying the item
        uses what the caller heard about it, and forgets it.
        """
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
        """The call that says ``batch``: the result wording's, or the fixed lines as written when none is worded."""
        if any(item.worded for item in batch):
            return self.wording.call(self, batch)
        line = self.delivered(batch, self.joined(batch, ""))
        return self.saying(line) if line else None

    def relay(self, batch: list[Owed]) -> TalkerCall | None:
        """The call that says ``batch``: its line worded early while its stamp holds, with no request; else a relay."""
        early, self.early = self.early, None
        if early is not None:
            early.cut()
            if early.words is not None and early.stamp == self._stamp(batch):
                self.observe("answer.worded_early", items=len(batch))
                line = self.delivered(batch, self.joined(batch, early.words))
                return self.saying(line) if line else None
        return self._relay_call(batch, self.conversation())

    def _relay_call(self, batch: Sequence[Owed], history: list[dict]) -> TalkerCall:
        """The relay that words ``batch``: the writer's frame and rules, ``history``, then the update."""
        content, note = self._update(batch)
        messages = self.layout.relay("\n\n".join(self.instructions), history, content, note)
        plan = Relay(tuple(batch), content, history, note, self.version)
        return TalkerCall(spec=self.answer_spec, messages=messages, format=self.talker.response_format, plan=plan)

    def _update(self, batch: Sequence[Owed]) -> tuple[str, str]:
        """The update a relay of ``batch`` words: its items' text, and its note on what the caller already heard."""
        worded = [item for item in batch if item.worded]
        delegations = list(dict.fromkeys(item.said.delegation_id for item in worded if item.said.delegation_id))
        told = " ".join(text for each in delegations if (text := self.spoken.told_text(each)))
        note = self.layout.note(told or None, self.worknote.newer_requests(delegations))
        return "\n\n".join(item.said.speak for item in worded), note

    # ------------------------------------------------------------------------------ the early wording
    def _stamp(self, batch: Sequence[Owed]) -> tuple:
        """What a wording of ``batch`` stands on: its items, its update's note, and the voice's version."""
        return tuple(batch), self._update(batch)[1], self.version

    def _deciding(self) -> bool:
        """Whether a caller turn or an instruction waits for its routing call, or is being decided."""
        return self.instruction_due or bool(self._undecided())

    def _word_early(self) -> None:
        """Word what is owed now, out of band, unless a wording of it stands; see the module docstring."""
        if self.talker is None or self._closed or not self.wording.words_early:
            return
        batch = tuple(entry for entry in self.owed if isinstance(entry, Owed))
        early = self.early
        if not any(item.worded for item in batch):
            return
        stamp = self._stamp(batch)
        if early is not None:
            if early.stamp == stamp:
                return
            early.cut()  # a move cuts a wording under way short
            if self._deciding():
                return  # worded again once no caller turn waits or is decided
        if self.waiting_lines:
            return  # they go to speech in the next run, before what is owed: worded once they are in flight
        # The lines in flight read as said, as they will be once they play.
        in_flight = [{"role": SPOKEN.role, "content": line.text} for line in self.spoken.lines.values()]
        call = self._relay_call(batch, [*self.conversation(), *in_flight])
        self.early = EarlyLine(batch, stamp)
        self.early.task = asyncio.get_running_loop().create_task(self._words(self.early, call), name="early-wording")

    async def _words(self, early: EarlyLine, call: TalkerCall) -> None:
        """Word ``early`` with ``call``, a relay run out of band; a failure holds the items as written."""
        try:
            async with asyncio.timeout(self.config.reliability.talker_attempt_timeout_seconds):
                reply = await self.talker.run_call(call)
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
        """A call that says ``line`` with no model call."""
        return TalkerCall(text=line.text, plan=Saying(line))

    @staticmethod
    def joined(batch: Sequence[Owed], words: str) -> str:
        """The batch's line: the worded items' words at the first one's place, the fixed lines as written."""
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
        """The line that says ``batch``: what it told about its delegations was worded in, and its items are said."""
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
        """A progress line about the quiet work offered, worded from the work note, if it is still quiet."""
        entry, self.progress_due = self.progress_due, None
        if not self.worknote.quiet(entry.id):
            return None
        update = self.worknote.offer(entry.id)
        self.observe("progress.offered", delegation_id=entry.id)
        note = self.layout.note(self.spoken.told_text(entry.id))
        messages = self.layout.progress("\n\n".join(self.instructions), self.conversation(), update, note)
        return TalkerCall(spec=self.progress_spec, messages=messages, plan=Progress(entry, self.version))

    def speaking(self, call: TalkerCall, text: str) -> None:
        """A routing ``speak``'s words went on to speech: they are this response's line."""
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
        """A reply: streamed already, or let out now (the holding line, speech decoded before the action)."""
        line = self.spoken.lines.get(self.responses) if call is not None and call.said else None
        if line is not None:
            if not decision.diagnostics.get("recovery"):  # a fallback keeps the words that went out
                line.text = decision.speech  # the whole line, for its resume, its count and the heard log
                self.heard_log.replace_agent(line.number, line.text)
            self.bound()
            self._move()
            return ""
        reply = Line(LineKind.REPLY, decision.speech)
        if call is None:  # the fallback after a failure: said in a run of its own
            self.say(reply)
            return ""
        return self.hand(reply)

    def _delegated(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """The turn goes to the delegation part, the decision's speech as its lead-in."""
        self.handoffs.delegate(plan.request, decision.speech, decided_at=self.floor.caller_starts)
        return ""

    def _waited(self, plan: Routing, call: TalkerCall | None, decision: TalkerDecision) -> str:
        """Nothing is said for the turn; a line its words cut says its unheard rest once."""
        self.observe("talker.waited")
        if plan.cut is None or not self.config.speech.resume_after_noise:
            return ""
        rest = self.spoken.rest(plan.cut)
        if rest is None:
            return ""
        self.observe("line.resumed")
        return self.hand(rest)

    def decided_by(self, plan: Routing) -> None:
        """The call's caller turns are decided, and an instruction it read with them."""
        for message in plan.covers:
            self.decided[id(message)] = message
        self.instruction_due = False

    def hand(self, line: Line) -> str:
        """``line`` goes to speech as the response in progress; return its words.

        The heard log notes it now, so a delegation made while it plays carries it, and its playout corrects that entry
        (:meth:`heard`, :meth:`not_heard`); a line said once more keeps the entry its first hand-off noted. The
        developer messages of the items it says go with it, right before its words (:meth:`written`), and the context
        is bounded with the line counted whole (:meth:`bound`).
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
        """The items ``line`` says are on their way into the context now, out of where a line nobody heard left them."""
        self._writes = [item.message for item in line.items if item.worded]
        ids = {id(message) for message in self._writes}
        if not ids:
            return
        if self.context is not None and any(id(message) in ids for message in self.context.get_messages()):
            self.context.set_messages([message for message in self.context.get_messages() if id(message) not in ids])
        self.sent = [message for message in self.sent if id(message) not in ids] + self._writes

    def written(self) -> list[dict]:
        """The developer messages that go into the context right before the line just handed: the items it says."""
        writes, self._writes = self._writes, []
        return writes

    def dropped(self, call: TalkerCall) -> None:
        """A result arrived before the call said anything."""
        self.call = None
        call.plan.dropped(self)
        self._after_call()

    def cancelled(self, call: TalkerCall) -> None:
        """An interruption cancelled the call (its plan's)."""
        if self.call is call:
            self.call = None
        call.plan.cancelled(self, call)
        self._after_call()

    def failed(self, call: TalkerCall, error: BaseException) -> None:
        """The call failed: its fallback runs from the pipeline's error event (:meth:`recover`)."""
        if self.call is call:
            self.call = None
        logger.warning(f"Talker failed ({type(error).__name__}: {error}); recovering the call")
        self.observe("talker.recovered", error_type=type(error).__name__)
        if call.plan.failed(self, call):
            self.failures.append(call)
        self._after_call()

    def recover(self) -> None:
        """Pipecat reported a failed call: each failed call's fallback (its plan's)."""
        while self.failures:
            self.failures.popleft().plan.recover(self)
        self._after_call()

    def response_started(self) -> int:
        """The talker starts a response: its number."""
        self.responses += 1
        return self.responses

    def response_ended(self) -> int:
        """The talker ends the response in progress: its number."""
        return self.responses

    def interrupted(self) -> None:
        """An interruption passed the talker: it drops the responses started so far that have not played."""
        self.spoken.interrupted(self.responses, on_purpose=self._on_purpose)
        self._on_purpose = False
        self._drop_retries()

    def _drop_retries(self) -> None:
        """A line waiting to be said once more, after it produced no audio, goes with the interruption: not heard."""
        retries = [line for line in self.waiting_lines if line.retried]
        if retries:
            self.waiting_lines = deque(line for line in self.waiting_lines if not line.retried)
            for line in retries:
                line.heard = ""
                self.not_heard(line)

    def caller_started(self) -> None:
        """The caller started speaking: lead-ins not said yet go, and the voice says nothing on its own meanwhile."""
        self.floor.caller(True)
        self.acknowledgements.drop_all()
        self._withdraw()
        self._moved()

    def caller_stopped(self) -> None:
        """The caller stopped speaking."""
        self.floor.caller(False)
        self._moved()

    def result_arrived(self) -> None:
        """A result arrived: the call in progress that has said nothing is cut, when a result changes what it read."""
        call = self.call
        if call is not None and not call.said and call.plan.cut_by_result():
            call.cut = True

    def _after_call(self) -> None:
        """A call ended: anything still due gets one more run once nobody speaks."""
        self._moved()
        if self._due() and not self.run_requested:
            self.request_run()

    def _due(self) -> bool:
        """Whether anything waits for a run of the talker."""
        owed = any(isinstance(entry, Owed) for entry in self.owed)
        return bool(owed or self.waiting_lines or self.progress_due or self.instruction_due or self._undecided())

    def _moved(self) -> None:
        call = self.call
        if call is not None and not call.said and call.plan.moved(self):
            call.cut = True  # a move cuts a wording under way short
        idle = not self.floor.caller_speaking and self.call is None and not self.spoken.in_flight() and not self._due()
        self.floor.settle(idle)
        self._word_early()

    def _move(self) -> None:
        """Something every wording reads moved: a line said or taken back, an instruction, thinking, a cut."""
        self.version += 1
        self._moved()

    # ------------------------------------------------------------------------------ the lines (spoken.Settled)
    def heard(self, line: Line, cut: bool) -> None:
        """The caller heard ``line``, even in part: a cut line keeps what played, and what it delivered is said."""
        if cut:
            self.heard_log.cut(line.number, line.heard)
        self._done(line.items)
        self._moved()

    def not_heard(self, line: Line) -> None:
        """Nobody heard ``line``: it leaves the heard log; what it delivered goes back to the front of what is owed."""
        self.heard_log.unsay(line.number)
        if line.items:
            self.owe_again(line.items)
            self.observe("answer.requeued", line=line.number, items=len(line.items))
            self.request_run()
        self._move()

    def say_again(self, line: Line) -> None:
        """``line`` produced no audio: it is said once more, behind the lines already waiting."""
        self.say(line)

    def owe_again(self, items: Sequence[Owed]) -> None:
        """``items`` are owed again, first."""
        for item in reversed(items):
            self.owed.appendleft(item)
            if item.worded:
                self.unsaid.add(id(item.message))

    def _done(self, items: Sequence[Owed]) -> None:
        """``items`` are not said again: the work note counts each result among them as said."""
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
        """Take back the lead-ins not said yet, as the caller's words would cut them: a later decision answers them."""
        withdrawn, self.lead_ins = list(self.lead_ins.values()), {}
        self.waiting_lines = deque(line for line in self.waiting_lines if all(line is not w for w in withdrawn))
        for line in withdrawn:
            self.observe("acknowledgement.dropped", delegation_id=line.delegation_ids[0])

    # ------------------------------------------------------------------------------ what the backend sends
    def _own_id(self, kind: str) -> str:
        self._made += 1
        return f"{kind}_{self._made}"

    def _owe(self, entry: Owed | Turnover) -> None:
        """Owe ``entry``; an end or a handover first in line applies now, unless the call under way words its item."""
        if self._closed:
            return
        self.owed.append(entry)
        while self.owed and isinstance(head := self.owed[0], Turnover) and not self._holds(head.delegation_id):
            self._turnover(self.owed.popleft())
        self._moved()

    def _holds(self, delegation_id: str) -> bool:
        """Whether the call under way words an item of ``delegation_id``: the next take skips its end or handover."""
        return self.call is not None and self.call.plan.holds(delegation_id)

    def ready(self, result: ReadyResult) -> None:
        """A ready result (a ``final_answer`` item): owed, and run once nobody speaks; its line writes it in."""
        message = {"role": "developer", "content": self.prompts.result_label + result.speak}
        self._register(message, RESULT)
        self.unsaid.add(id(message))
        self._owe(Owed(result, LineKind.RESULT, message, answers=True))
        self.request_run()

    def notice(self, delegation_id: str | None, text: str) -> None:
        """A fixed line, said word for word in its turn; an empty line says nothing."""
        if text.strip():
            self._owe(Owed(ReadyResult(delegation_id or "", self._own_id("notice"), text), LineKind.NOTICE))
            self.request_run()

    def ended(self, delegation_id: str) -> None:
        """The delegation ended with nothing more to say: in its turn, what the caller heard is forgotten."""
        self._owe(Turnover(delegation_id))

    def handed_over(self, delegation_id: str, taker: str) -> None:
        """The delegation's work went to ``taker``: in its turn, what the caller heard goes with the taker."""
        self._owe(Turnover(delegation_id, taker))

    def commentary(self, text: str, delegation_id: str | None = None) -> None:
        """The application's commentary (a fact to say); ``delegation_id``: the client delegation it answers."""
        self.acknowledgements.drop_all()
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
        """Silent context for later decisions and for the next delegation; nothing is spoken."""
        self.acknowledgements.drop_all()
        self.heard_log.context(text)
        self._append({"role": "developer", "content": self.prompts.context_label + text}, APP_CONTEXT, run=False)
        self._move()

    def typed(self, text: str) -> None:
        """The application typed a question itself: the talker's context holds it; the heard log does not."""
        self._append({"role": "developer", "content": self.prompts.context_label + text}, TYPED, run=False)

    async def instruction(self, text: str) -> None:
        """Steer behavior: pin the direction, cut the line in progress for good, and decide it in turn."""
        self.instructions.append(text)
        self.heard_log.instruction(text)
        self.instruction_due = True
        self._on_purpose = True
        self._move()
        if self.talker is not None:
            # An interruption at the talker, not at the pipeline's start: the caller's turn in progress stays.
            await self.talker.queue_frame(InterruptionFrame())
        self.request_run()

    # ------------------------------------------------------------------------------ Pipecat's events
    def attach(
        self,
        user: LLMUserAggregator,
        assistant: LLMAssistantAggregator,
        worker: PipelineWorker,
    ) -> None:
        """Hear the pipeline's events: each assistant turn's end, the caller's idle spells, and its errors.

        At each assistant turn's end, a line an interruption cut gets the cut mark, and the context is bounded
        (:meth:`bound`). The voice's fallback for a failed call runs from the pipeline's error event.
        """

        @assistant.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(aggregator, message):
            if message.interrupted:
                self.cut_marked(aggregator.context, message.content)
            self.bound()

        @user.event_handler("on_user_turn_idle")
        async def on_user_turn_idle(aggregator):
            self.on_user_idle()

        @worker.event_handler("on_pipeline_error")
        async def on_pipeline_error(worker, frame):
            if frame.processor is self.talker:
                self.recover()

    def on_user_idle(self) -> None:
        """The caller has been idle since a turn with no words, or since the bot spoke: run whatever is still due.

        A turn with no words brings no run, so a run the assistant aggregator owes for it is asked for again.
        """
        if self._due():
            self.run_requested = False
            self.request_run()

    def cut_marked(self, context: LLMContext, played: str) -> None:
        """A line was cut after ``played``: the context's record of it ends with the cut mark.

        With ``speech.heard_words: estimated``, it keeps the words its audio likely carried past what played. A
        stitched line reports its words only at its end: one cut while it played is kept whole, as the heard log keeps
        it, or, with ``estimated``, as the words its audio likely carried.
        """
        kept = self.spoken.kept(played)
        if not kept:
            return
        if not played:
            context.add_message({"role": "assistant", "content": kept.rstrip() + self.prompts.cut_mark})
        else:
            message = next((each for each in reversed(context.get_messages()) if each.get("role") == "assistant"), None)
            if message is None or message_text(message) != played:
                return
            message["content"] = kept.rstrip() + self.prompts.cut_mark
        self._move()

    async def _quiet(self, after: float) -> None:
        """Offer a progress line when work has run through a quiet spell; see the module docstring."""
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
