# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation turn manager: three lanes decoupled in time (plan section 4.1).

* **Frontend lane** -- every committed user turn gets one ``delegate`` decision.
  Speech before the decision returns cancels it and merges the words into the next
  transcript (``cancel_and_merge``); once a delegation is sent it stands, and the next
  utterance becomes a steer (the gateway decides).
* **Backend lane** -- the gateway pushes tool calls, answers, status reports and state
  changes whenever they are ready; nothing here waits for the backend.
* **Output lane** -- one response at a time, chosen by :class:`OutputQueue`; audio is
  never released while the user speaks, and every spoken item gets a playback outcome
  (heard / partial / not heard) that the shared transcript records (plan 7.4).

tau3-failure-fixes-plan.md adds, each behind its own switch (all off by default):
M1 one proactive status line per run while the backend works in silence
(``output.proactive_status``); M3.2 a spelling hold before deciding a turn that ends
mid-spelling (``delegation.spelling_hold``); M2 replaying an unheard answer on an
explicit "status" request in IDLE (``delegation.replay_unheard_answer``, RC2); G1 filler
de-duplication (``delegation.filler_dedupe``); G2 answer cleanup (``output.clean_answers``).

fix-plan-f1-f2-f4-f8.md adds, also off by default: F2 rules R1-R3 that never leave a liveness ping or a
question silent (``delegation.guards``), and F1 replay triggers that speak an unheard answer again when the
user only backchannelled, pinged or asked to repeat (``delegation.replay_unheard_answer.triggers``).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from examples.frontend_delegation_hermes.core.agent_tools import capability_lines, realtime_tools_to_specs
from examples.frontend_delegation_hermes.core.backend.protocol import (
    IDLE,
    NO_SESSION,
    REQUEST_STATUS,
    REQUEST_TASK,
    VOICE_TO_GATEWAY,
    WORKING,
    message,
)
from examples.frontend_delegation_hermes.core.config import DelegationConfig
from examples.frontend_delegation_hermes.core.engine.backend_link import BackendLink
from examples.frontend_delegation_hermes.core.engine.filler import Stamp
from examples.frontend_delegation_hermes.core.engine.output_scheduler import (
    CallsItem,
    Dropped,
    OutputQueue,
    SpeechItem,
)
from examples.frontend_delegation_hermes.core.engine.playback import ResponseProgress
from examples.frontend_delegation_hermes.core.engine.spelling_hold import SpellingHoldPredicate
from examples.frontend_delegation_hermes.core.engine.spoken_text import clean_for_speech
from examples.frontend_delegation_hermes.core.engine.transcript import (
    HEARD,
    NOT_HEARD,
    PARTIAL,
    Entry,
    SharedTranscript,
)
from examples.frontend_delegation_hermes.core.errors import WireProtocolError
from examples.frontend_delegation_hermes.core.frontend.decider import (
    LIVENESS_REPLY,
    PING_AS_STATUS,
    QUESTION_NEVER_SILENT,
    Decider,
    DecisionContext,
    apply_guards,
)
from examples.frontend_delegation_hermes.core.frontend.delegate_tool import Decision
from examples.frontend_delegation_hermes.core.frontend.prompts import PromptCatalog
from examples.frontend_delegation_hermes.core.frontend.status_verbalizer import StatusVerbalizer
from examples.frontend_delegation_hermes.core.frontend.turn_class import (
    BACKCHANNEL,
    LIVENESS,
    TurnClass,
    classify_turn,
    lexicon_for,
)
from examples.frontend_delegation_hermes.core.head_api import SessionChange, TurnContext, UserInput
from examples.frontend_delegation_hermes.core.normalization.arguments import ArgumentNormalizer
from examples.frontend_delegation_hermes.core.normalization.schema_rules import SchemaRules, tools_sha256
from examples.frontend_delegation_hermes.core.normalization.transcript import TranscriptNormalizer
from examples.frontend_delegation_hermes.core.prompt_features import sha256_text, tools_canonical_sha256
from examples.frontend_delegation_hermes.core.text.tools import ToolSpec
from examples.frontend_delegation_hermes.core.tools.field_provenance import FieldProvenance, SpelledSpan
from examples.frontend_delegation_hermes.core.tools.format_retry import FormatRetry
from examples.frontend_delegation_hermes.core.tools.relay import FormatRetryMessages, ToolRelay
from examples.frontend_delegation_hermes.core.tools.result_hints import ResultHints
from examples.frontend_delegation_hermes.core.wire import server_events as ev
from examples.frontend_delegation_hermes.core.wire.response_writer import ResponseWriter
from examples.frontend_delegation_hermes.core.wire.session_view import SessionSettings


@dataclass(frozen=True, slots=True)
class SessionRules:
    """The tool-dependent collaborators, rebuilt once the client's tools are known."""

    argument_normalizer: ArgumentNormalizer | None = None
    result_hints: ResultHints | None = None
    spelling_hold: SpellingHoldPredicate | None = None
    schema: SchemaRules | None = None
    format_retry: FormatRetry | None = None


@dataclass(slots=True)
class SessionParts:
    """Per-session collaborators built by the factory (all swappable, plan 12.1).

    The tool-dependent ones (normalizer, result hints, spelling hold) start without schema
    rules; ``rebuild`` (set when a config switch is ``auto``) derives them from the tools of
    each applied ``session.update``.
    """

    decider: Decider
    verbalizer: StatusVerbalizer
    link: BackendLink
    prompts: PromptCatalog
    transcript_normalizer: TranscriptNormalizer | None = None
    argument_normalizer: ArgumentNormalizer | None = None
    result_hints: ResultHints | None = None
    local_tools: tuple[ToolSpec, ...] = ()
    spelling_hold: SpellingHoldPredicate | None = None
    rebuild: Callable[[Sequence[Mapping[str, Any]]], SessionRules] | None = None
    format_retry: FormatRetry | None = None
    format_messages: FormatRetryMessages | None = None


@dataclass(slots=True)
class _Turn:
    turn_id: int
    text: str
    turn_start: Stamp | None = None
    turn_end: Stamp | None = None
    asr_final: Stamp | None = None
    decided: Stamp | None = None
    first_audio_logged: bool = False
    answer_audio_logged: bool = False
    #: Started by a client ``response.create`` (manual mode): the client waits for a ``response.done``.
    requested: bool = False


@dataclass(slots=True)
class _Running:
    """The response being sent now."""

    writer: ResponseWriter
    item: SpeechItem | CallsItem
    task: asyncio.Task[None] | None = None
    speak_task: asyncio.Task[None] | None = None
    item_id: str = ""
    interrupted: bool = False


@dataclass(slots=True)
class _Backend:
    """What the voice side knows about the backend (mirrored from gateway messages)."""

    state: str = NO_SESSION
    epoch: int = 0
    run_id: str = ""
    run_started: float | None = None
    current_request: str = ""
    activity: deque[str] = field(default_factory=lambda: deque(maxlen=6))
    unavailable: str = ""
    ready: asyncio.Future[dict[str, Any]] | None = None
    configured: asyncio.Future[dict[str, Any]] | None = None
    # M2 state (tau3-failure-fixes-plan.md section 4).
    last_finished_run_id: str = ""  # set on backend_run_done
    last_finished_epoch: int = 0
    last_finished_request: str = ""  # the text of the run's first turn
    last_finished_turn_ids: tuple[int, ...] = ()
    last_task_turn_id: int = 0  # the newest turn delegated as a task


class DelegationTurnManager:
    """Implements ``TurnManagerLike`` for the frontend-delegation prototype."""

    def __init__(self, context: TurnContext, config: DelegationConfig, parts: SessionParts) -> None:
        """Bind the session; nothing runs until :meth:`start`."""
        self._ctx = context
        self._config = config
        self._parts = parts
        self._voice = context.config
        self.progress = ResponseProgress()
        self.transcript = SharedTranscript()
        self._queue = OutputQueue(
            hold_max_ms=config.output.hold_max_ms, on_stale=config.output.on_stale, monotonic=context.clock.monotonic
        )
        per_call_delay = config.backend.delay_seconds if config.backend.delay_where == "per_tool_call" else 0.0
        hints = context.config.normalization.tool_arguments.result_hints
        # A3: the decided user turns, for the field-provenance line of repeated lookup misses.
        self._provenance = FieldProvenance() if hints.enabled and hints.field_provenance else None
        self._relay = ToolRelay(
            executor=config.tools.executor,
            send_result=self._send_tool_result,
            emit_calls=self._push,
            log=self._log,
            normalizer=parts.argument_normalizer,
            result_hints=parts.result_hints,
            local_tools=parts.local_tools,
            batch_window_ms=config.tools.batch_window_ms,
            per_call_delay_s=per_call_delay,
            format_retry=parts.format_retry,
            format_messages=parts.format_messages,
            provenance=self._provenance,
            result_timeout_s=config.tools.result_timeout_s,
        )
        self._backend = _Backend(activity=deque(maxlen=max(1, config.frontend.activity_max_items)))
        self._system_prompt = ""
        self._configured_with: tuple[list[dict[str, Any]], str] | None = None
        self._tools_sha256 = ""  # the tools the backend runs with (session_tools_sha256)
        self._pending_tools_sha256 = ""  # the tools of the session.configure in flight
        self._pending_tools_canonical = ""  # the same tools' tools_canonical_sha256 (joins the backend trace)
        self._turns: dict[int, _Turn] = {}
        self._turn_counter = 0
        self._awaiting = 0  # utterances started (speech_started) whose transcript has not arrived
        self._decision: asyncio.Task[None] | None = None
        self._decision_turn: _Turn | None = None
        self._merge_prefix = ""
        self._pending_inputs: list[UserInput] = []
        self._running: _Running | None = None
        self._unheard: dict[str, Entry] = {}
        self._wake = asyncio.Event()
        self._output_task: asyncio.Task[None] | None = None
        self._side_tasks: set[asyncio.Task[Any]] = set()
        self._greeted = False
        self._closed = False
        self._link_open = False
        # M1: proactive status while the backend works in silence.
        self._quiet_since: float | None = None
        self._proactive_epoch = 0
        self._proactive_count = 0
        self._proactive_lines = parts.prompts.lines("status_proactive_lines")
        self._proactive_next = 0
        self._proactive_task: asyncio.Task[None] | None = None
        # M3.2: the previous turn was merged during a spelling hold.
        self._hold_accumulator = False
        # G1: the session's recent delegated fillers (normalized).
        self._recent_fillers: deque[str] = deque(maxlen=max(1, config.delegation.filler_dedupe.recent))
        self._filler_alternatives = parts.prompts.lines("filler_alternatives")
        # F1/F2 (fix-plan-f1-f2-f4-f8.md section 2): the session's phrase lists, chosen by the ASR language.
        language = self._voice.asr.endpoint.language_code
        self._lexicon_name, self._lexicon = lexicon_for(language, config.delegation.liveness.lexicons)
        self._lexicon_missing_logged = False
        self._liveness_lines = (
            parts.prompts.lines("liveness_lines") if config.delegation.liveness.liveness_reply else []
        )
        self._liveness_next = 0
        self._alternative_used: dict[str, int] = {}  # filler alternative -> the turn that last spoke it

    # -- TurnManagerLike: lifecycle ------------------------------------------------------------

    @property
    def state(self) -> str:
        """Frontend/output state plus the backend's."""
        if self._decision is not None and not self._decision.done():
            lane = "DECIDING"
        elif self._running is not None:
            lane = "SPEAKING"
        else:
            lane = "LISTENING"
        return f"{lane}/{self._backend.state}"

    async def start(self) -> None:
        """Connect to the gateway (the worker starts while the client sends session.update)."""
        loop = asyncio.get_running_loop()
        self._backend.ready = loop.create_future()
        self._output_task = asyncio.create_task(self._output_loop(), name=f"output-{self._ctx.session_id}")
        seed = self._voice.protocol.seed_history_with_client_greeting
        if seed:
            self.transcript.seed_greeting(seed)
        arguments = self._voice.normalization.tool_arguments
        hints_on = arguments.enabled and arguments.result_hints.enabled
        self._log(
            "fdh_session_start",
            config_hash=self._config.config_hash,
            config=self._config.effective,
            model=self._ctx.model,
            features=self._config.features,
            invalid_message_keys={
                "invalid": arguments.invalid_message_key if arguments.enabled else "",
                "escalate_invalid": arguments.escalate_invalid_message_key if arguments.enabled else "",
                "already_failed": arguments.retry_guard.message_key
                if arguments.enabled and arguments.retry_guard.enabled
                else "",
                "result_hint": arguments.result_hints.message_key if hints_on else "",
                "result_hint_escalate": arguments.result_hints.escalate_message_key if hints_on else "",
            },
        )
        if self._config.output.proactive_status.enabled:
            self._proactive_task = asyncio.create_task(
                self._proactive_loop(), name=f"proactive-status-{self._ctx.session_id}"
            )
        try:
            await self._parts.link.open(self._on_backend_message)
        except Exception as exc:  # noqa: BLE001 - reported on session.update, never crashes the session
            self._backend.unavailable = f"cannot reach the backend gateway: {exc}"
            logger.error(f"[{self._ctx.session_id}] {self._backend.unavailable}")
            self._log("sidecar_link", event="connect_failed", error=str(exc))
            return
        self._link_open = True
        self._log("sidecar_link", event="connected")
        be = self._config.backend
        self._send(
            "session.open",
            session_id=self._ctx.session_id,
            settings={
                "simulated_delay": {"seconds": be.delay_seconds, "where": be.delay_where},
                "steer_mode": be.steer_mode,
                "update_after_start": self._config.update_after_start,
            },
        )

    async def close(self) -> None:
        """Stop every lane, tell the gateway, record what was never heard."""
        if self._closed:
            return
        self._closed = True
        tasks: list[asyncio.Task[Any] | None] = [
            self._decision,
            self._output_task,
            self._proactive_task,
            *self._side_tasks,
        ]
        if self._running is not None:
            tasks.extend([self._running.speak_task, self._running.task])
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
        for task in tasks:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for item in self._queue.drain():
            self._settle(item.entry, NOT_HEARD, reason="session_closed")
        for entry in list(self._unheard.values()):
            self._settle(entry, NOT_HEARD, reason="session_closed")
        self._unheard.clear()
        await self._relay.close()
        if self._link_open:
            with contextlib.suppress(Exception):
                self._send("session.close")
        with contextlib.suppress(Exception):
            await self._parts.link.close()

    # -- TurnManagerLike: session configuration ------------------------------------------------

    async def on_session_update(self, settings: SessionSettings, change: SessionChange) -> None:
        """Configure the backend (transactional: raising rejects the update)."""
        tools = self._tool_dicts(settings)
        policy = settings.instructions if "backend" in self._config.instructions_apply_to else ""
        system_prompt = self._frontend_system_prompt(tools, settings.instructions)
        sent = (tools, policy)
        if change.first or self._backend.configured is None or sent != self._configured_with:
            # Only a real change reaches the gateway: a browser re-sending the same (or ignored) tools
            # must not trip session.update_after_start.
            self._pending_tools_sha256 = tools_sha256(tools)
            self._pending_tools_canonical = tools_canonical_sha256(tools)
            configured = await self._configure(tools, policy)
            self._configured_with = sent
            if configured.get("applied", True):
                self._rebuild_session_rules(tools)
        if system_prompt != self._system_prompt:
            self._log(
                "frontend_prompt",
                frontend_prompt_sha256=sha256_text(system_prompt),
                prompt_features=dict(self._config.prompt_features),
            )
        self._system_prompt = system_prompt

    def _rebuild_session_rules(self, tools: list[dict[str, Any]]) -> None:
        """Derive the tool-dependent collaborators from the tools the backend now runs with."""
        rebuild = self._parts.rebuild
        if rebuild is None:
            return
        rules = rebuild(tools)
        self._parts.argument_normalizer = rules.argument_normalizer
        self._parts.result_hints = rules.result_hints
        self._parts.spelling_hold = rules.spelling_hold
        self._parts.format_retry = rules.format_retry
        self._relay.reconfigure(rules.argument_normalizer, rules.result_hints, rules.format_retry)
        ledger = rules.result_hints.ledger if rules.result_hints is not None else None
        self._log(
            "session_rules",
            session_tools_sha256=self._tools_sha256,
            schema_auto=self._config.schema_auto,
            **(rules.schema.snapshot() if rules.schema is not None else {}),
            **({"identification_tools": list(ledger.identification)} if ledger is not None else {}),
        )

    async def _configure(self, tools: list[dict[str, Any]], instructions: str) -> dict[str, Any]:
        if self._backend.unavailable:
            raise WireProtocolError(self._backend.unavailable, code="backend_unavailable", param="session")
        timeout = self._config.backend.open_timeout_s
        ready = self._backend.ready
        if ready is not None and not ready.done():
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=timeout)
            except TimeoutError as exc:
                raise WireProtocolError(
                    f"the backend worker was not ready within {timeout:g}s", code="worker_start_timeout"
                ) from exc
        if ready is not None and ready.done() and ready.exception() is not None:
            raise WireProtocolError(str(ready.exception()), code="backend_unavailable", param="session")
        self._backend.configured = asyncio.get_running_loop().create_future()
        self._send("session.configure", tools=tools, instructions=instructions)
        try:
            return await asyncio.wait_for(asyncio.shield(self._backend.configured), timeout=timeout)
        except TimeoutError as exc:
            raise WireProtocolError(
                f"the backend was not configured within {timeout:g}s", code="backend_timeout"
            ) from exc
        except _GatewayError as exc:
            raise WireProtocolError(exc.text, code=exc.code, param="session") from exc

    def _tool_dicts(self, settings: SessionSettings) -> list[dict[str, Any]]:
        if self._voice.tools.source == "client":
            return [dict(tool) for tool in settings.tools]
        return [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": dict(spec.parameters),
            }
            for spec in self._parts.local_tools
        ]

    def _frontend_system_prompt(self, tools: list[dict[str, Any]], instructions: str) -> str:
        capabilities = list(capability_lines(realtime_tools_to_specs(tools))) if tools else []
        note = ""
        if self._parts.transcript_normalizer is not None:
            note = self._parts.prompts.render("normalization_note")
        prompt = self._parts.prompts.render("frontend_system", capabilities=capabilities, normalization_note=note)
        if instructions.strip() and "frontend" in self._config.instructions_apply_to:
            prompt = f"{prompt}\n\nService policy (for context only; the backend applies it):\n{instructions.strip()}"
        return prompt

    # -- TurnManagerLike: input ----------------------------------------------------------------

    def on_audio_advance(self, step_ms: float) -> None:
        """Move the playout cursor and record fully heard items."""
        self.progress.advance(step_ms)
        if not self._unheard:
            return
        running_id = self._running.item_id if self._running is not None else ""
        for item_id, entry in list(self._unheard.items()):
            span = self.progress.item(item_id)
            if item_id == running_id or span is None:
                continue
            if span.end_ms <= self.progress.heard_ms:
                del self._unheard[item_id]
                self._settle(entry, HEARD)

    async def on_speech_started(self) -> None:
        """Confirmed user speech: hold output, cancel an undecided turn, barge in on audio."""
        self._awaiting += 1
        self._mark_activity()
        if self._decision is not None and not self._decision.done() and self._decision_turn is not None:
            turn = self._decision_turn
            self._decision.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._decision
            self._merge_prefix = f"{self._merge_prefix} {turn.text}".strip()
            self._log("decision_cancelled", turn_id=turn.turn_id, reason="turn_detected", merge=True)
        settings = self._ctx.settings()
        if self._voice.barge_in.enabled and settings.turn_detection.interrupt_response and self.progress.active:
            await self._interrupt_output(reason="turn_detected")
        self._wake.set()

    def on_user_input(self, user_input: UserInput, *, from_audio: bool) -> None:
        """A committed utterance or user text item."""
        if from_audio:
            self._awaiting = max(0, self._awaiting - 1)
        text = user_input.text.strip()
        if self._merge_prefix:
            text = f"{self._merge_prefix} {text}".strip()
            self._merge_prefix = ""
        if len(text) < max(1, self._voice.turn_detection.min_transcript_chars):
            self._log("empty_transcript", text=text)
            self._wake.set()
            return
        user_input.text = text
        auto = self._voice.protocol.auto_response and self._ctx.settings().turn_detection.create_response
        if not (from_audio and auto):
            self._pending_inputs.append(user_input)
            self._wake.set()
            return
        self._start_decision(user_input)

    def on_utterance_dropped(self, item_id: str, *, reason: str) -> None:
        """An utterance ended without a transcript."""
        self._awaiting = max(0, self._awaiting - 1)
        self._log("utterance_dropped", item_id=item_id, reason=reason)
        self._wake.set()

    def on_function_output(self, call_id: str, output: str) -> None:
        """Client ``function_call_output``."""
        self._relay.on_function_output(call_id, output)

    def on_response_create(self) -> None:
        """Client ``response.create``, in the existing manager's order (plan 8.2)."""
        if self._relay.take_ack():
            return
        if self._deciding() or self._running is not None:
            raise WireProtocolError(
                "a response is already in progress", code="conversation_already_has_active_response", param=None
            )
        if self._pending_inputs:
            inputs, self._pending_inputs = self._pending_inputs, []
            self._start_decision(
                UserInput(
                    text=" ".join(item.text for item in inputs),
                    turn_start=inputs[0].turn_start,
                    turn_end=inputs[-1].turn_end,
                    asr_final=inputs[-1].asr_final,
                ),
                requested=True,
            )
            return
        protocol = self._voice.protocol
        if protocol.greeting_enabled and not self._greeted and protocol.greeting_text:
            self.speak_greeting()
            return
        raise WireProtocolError(
            "response.create with no new user input to respond to", code="invalid_value", param=None
        )

    def delete_pending_input(self, item_id: str) -> bool:
        """Drop an unconsumed user text item (manual-response mode)."""
        for index, item in enumerate(self._pending_inputs):
            if item.item_id == item_id:
                del self._pending_inputs[index]
                return True
        return False

    def on_truncate(self, item_id: str, content_index: int, audio_end_ms: int) -> None:
        """Client truncate: acknowledge with a clamped time and record what was heard."""
        span = self.progress.item(item_id)
        if span is None:
            if item_id not in self._ctx.conversation.item_ids:
                raise WireProtocolError(f"unknown item_id {item_id!r}", code="invalid_value", param="item_id")
            self._ctx.emit(ev.item_truncated(item_id, content_index, 0))
            return
        duration = max(0.0, span.end_ms - span.start_ms)
        clamped = int(max(0.0, min(float(audio_end_ms), duration)))
        self._log("truncate", item_id=item_id, client_audio_end_ms=audio_end_ms, clamped_ms=clamped)
        entry = self._unheard.pop(item_id, None)
        if entry is not None:
            heard = self.progress.heard_text(item_id)
            self._settle(entry, HEARD if heard == " ".join(entry.text.split()) else PARTIAL, heard_text=heard)
        self._ctx.emit(ev.item_truncated(item_id, content_index, clamped))

    def speak_greeting(self) -> None:
        """Speak the configured greeting as its own response."""
        text = self._voice.protocol.greeting_text
        if not text or self._greeted:
            return
        self._greeted = True
        entry = self.transcript.add_spoken("frontend_speech", "frontend", text)
        self._push(SpeechItem(kind="greeting", text=text, entry=entry))

    async def on_response_cancel(self) -> None:
        """Client ``response.cancel``: drop an undecided turn, or stop the audio."""
        if self._deciding():
            assert self._decision is not None  # noqa: S101 - _deciding checks it
            self._decision.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._decision
            self._log("decision_cancelled", reason="client_cancelled", merge=False)
            return
        if self._running is not None:
            await self._interrupt_output(reason="client_cancelled")

    async def on_output_audio_clear(self) -> None:
        """``output_audio_buffer.clear``."""
        if self.progress.active:
            await self._interrupt_output(reason="client_cancelled")

    def on_filler(self, text: str | None, stamp: Stamp) -> None:
        """Unused: fillers come from the frontend decision, not from a text-agent sink."""

    # -- frontend lane --------------------------------------------------------------------------

    def _deciding(self) -> bool:
        return self._decision is not None and not self._decision.done()

    def _start_decision(self, user_input: UserInput, *, requested: bool = False) -> None:
        self._turn_counter += 1
        turn = _Turn(
            turn_id=self._turn_counter,
            text=user_input.text,
            turn_start=user_input.turn_start,
            turn_end=user_input.turn_end,
            asr_final=user_input.asr_final,
            requested=requested,
        )
        self._turns[turn.turn_id] = turn
        self._decision_turn = turn
        self._decision = asyncio.create_task(self._decide(turn), name=f"decide-{turn.turn_id}")

    async def _decide(self, turn: _Turn) -> None:
        await self._spelling_hold(turn)
        text, spelled = self._normalize_with_spans(turn.text)
        heard = self.transcript.last_heard_assistant()
        question = heard.spoken_text if heard is not None else ""
        backend = self._backend
        messages = self.transcript.frontend_messages(max_groups=self._config.frontend.history_max_groups)
        if messages and messages[-1]["role"] == "user":
            messages[-1] = {"role": "user", "content": f"{messages[-1]['content']} {text}"}
        else:
            messages.append({"role": "user", "content": text})
        elapsed = None
        if backend.state == WORKING and backend.run_started is not None:
            elapsed = int(self._ctx.clock.monotonic() - backend.run_started)
        context = DecisionContext(
            system_prompt=self._system_prompt or self._frontend_system_prompt([], ""),
            messages=messages,
            user_text=text,
            state=backend.state,
            current_request=backend.current_request if backend.state == WORKING else "",
            elapsed_s=elapsed,
            recent_activity=list(backend.activity),
            backend_asked_question=self.transcript.backend_asked_question(),
            last_answer_unheard=self._last_answer_unheard(),
        )
        decision = await self._parts.decider.decide(context)
        # Everything below is synchronous: a barge-in can no longer undo this decision.
        if self._provenance is not None:
            self._provenance.add_turn(turn.text, spelled, question)
        hints = self._parts.result_hints
        if hints is not None and hints.ledger is not None:
            hints.ledger.add_turn(turn.turn_id, turn.text, [span.written for span in spelled], question)
        turn_class = self._classify(text)
        liveness = self._config.delegation.liveness
        guarded = apply_guards(
            decision,
            context,
            backend_question_needs_delegate=self._config.delegation.backend_question_needs_delegate,
            backchannel_words=self._config.delegation.backchannel_words,
            turn_class=turn_class,
            liveness_reply=liveness.liveness_reply,
            question_never_silent=liveness.question_never_silent,
            ping_as_status=liveness.ping_as_status,
        )
        if guarded is not decision:
            details: dict[str, Any] = {}
            if turn_class is not None:
                details = {"turn_class": turn_class.kind, "backend_state": context.state}
            self._log("delegation_guard", turn_id=turn.turn_id, guard=guarded.repair, **details)
        trigger = self._replay_trigger(guarded, context, turn_class)
        if trigger and self._replay(turn, text, guarded, context, trigger=trigger):
            return
        self._apply(turn, text, guarded, context)

    def _classify(self, text: str) -> TurnClass | None:
        """The F1/F2 class of a turn, or None when no rule needs it (today's path)."""
        if not self._config.delegation.classifies_turns:
            return None
        if self._lexicon is None and not self._lexicon_missing_logged:
            self._lexicon_missing_logged = True
            self._log("lexicon_missing", language=self._voice.asr.endpoint.language_code)
        return classify_turn(text, self._lexicon)

    def _replay_trigger(self, decision: Decision, context: DecisionContext, turn_class: TurnClass | None) -> str:
        """The F1 trigger this decision matches ("" for none); see fix-plan-f1-f2-f4-f8.md section 2.4."""
        replay = self._config.delegation.replay
        if not replay.enabled:
            return ""
        triggers = replay.triggers
        if "status_intent" in triggers and decision.delegate and decision.request == REQUEST_STATUS:
            return "status_intent"  # M2: _replay checks the state
        if turn_class is None or context.state == WORKING:
            return ""
        silent = not decision.delegate and not decision.filler_text
        if turn_class.kind == BACKCHANNEL and silent and "backchannel" in triggers:
            return "backchannel"
        if turn_class.kind == LIVENESS and (silent or decision.repair == LIVENESS_REPLY):
            name = "repeat" if turn_class.repeat else "ping"
            if name in triggers:
                return name
        return ""

    async def _spelling_hold(self, turn: _Turn) -> None:
        """M3.2: wait ``hold_ms`` before deciding a turn that ends mid-spelling (speech merges it)."""
        predicate = self._parts.spelling_hold
        accumulator, self._hold_accumulator = self._hold_accumulator, False
        if predicate is None or turn.requested:
            return
        check = predicate.check(turn.text, accumulator=accumulator)
        if not check.hold:
            return
        hold_ms = self._config.delegation.spelling_hold.hold_ms
        details = {"turn_id": turn.turn_id, "hold_ms": hold_ms, "evidence": check.evidence, "value": check.value}
        try:
            await asyncio.sleep(hold_ms / 1000.0)
        except asyncio.CancelledError:
            # Speech started: on_speech_started merges this text into the next turn.
            self._hold_accumulator = True
            self._log("spelling_hold", **details, merged=True)
            raise
        self._log("spelling_hold", **details, merged=False)

    def _normalize_with_spans(self, text: str) -> tuple[str, tuple[SpelledSpan, ...]]:
        """The normalized text, and its ``spelled_runs`` spans (positions in ``text``) for A3."""
        normalizer = self._parts.transcript_normalizer
        if normalizer is None:
            return text, ()
        normalized = normalizer.normalize(text)
        if normalized.changed:
            spans = [{"spoken": span.spoken, "written": span.written, "kind": span.kind} for span in normalized.spans]
            self._log("transcript_normalized", raw=text, text=normalized.text, spans=spans)
        spelled = tuple(
            SpelledSpan(written=span.written, start=span.start, end=span.end)
            for span in normalized.spans
            if span.kind == "spelled"
        )
        return normalized.text, spelled

    # -- M2: replay an unheard answer (RC2, tau3-failure-fixes-plan.md section 4) --------------

    def _last_answer_unheard(self) -> bool:
        """The newest backend answer was not heard or only partly (shown only with ``replay_intent``)."""
        if not self._config.prompt_features.get("replay_intent"):
            return False
        entry = self.transcript.last_backend_answer()
        return entry is not None and entry.outcome in (NOT_HEARD, PARTIAL) and not entry.replayed

    def _replay(
        self, turn: _Turn, text: str, decision: Decision, context: DecisionContext, *, trigger: str = "status_intent"
    ) -> bool:
        """Speak the unheard answer again instead of the decision, when every guard passes."""
        if not self._config.delegation.replay.enabled:
            return False
        if trigger == "status_intent" and (
            not decision.delegate or decision.request != REQUEST_STATUS or context.state == WORKING
        ):
            return False  # no explicit intent in IDLE: today's path (RC1)
        entry = self.transcript.last_backend_answer()
        reason, replay_text = self._replay_check(entry)
        if reason or entry is None:
            self._log("answer_replay_skipped", turn_id=turn.turn_id, reason=reason or "no_answer", trigger=trigger)
            return False
        turn.decided = Stamp.now(self._ctx.clock, self._ctx.audio_now())
        self.transcript.add_user(text, turn_id=turn.turn_id, delegated=False)
        self._log(
            "delegation_decision",
            turn_id=turn.turn_id,
            delegate=decision.delegate,
            request=decision.request,
            filler=decision.filler_text,
            filler_chars=len(decision.filler_text),
            backend_state=context.state,
            latency_ms=int(decision.latency_ms),
            user_stop_to_decision_ms=_ms(turn.turn_end, turn.decided),
            usage=decision.usage,
            repair=decision.repair,
            text=text,
            replay=True,
        )
        self._flush_context()
        entry.replayed = True
        assert entry.released_mono is not None  # noqa: S101 - checked by _replay_check
        self._log(
            "answer_replayed",
            turn_id=turn.turn_id,
            answer_id=entry.answer_id,
            run_id=entry.run_id,
            request=self._backend.last_finished_request,
            outcome_before=entry.outcome,
            age_s=round(self._ctx.clock.monotonic() - entry.released_mono, 2),
            chars=len(replay_text),
            trigger=trigger,
        )
        spoken = self.transcript.add_spoken("frontend_speech", "frontend", replay_text, turn_id=turn.turn_id)
        self._push(
            SpeechItem(
                kind="replay",
                text=replay_text,
                entry=spoken,
                turn_id=turn.turn_id,
                usage=dict(decision.usage),
                on_first_audio=lambda: self._first_audio(turn, "filler"),
            )
        )
        return True

    def _replay_check(self, entry: Entry | None) -> tuple[str, str]:
        """``(reason, text)``: the first failed guard (empty when all pass) and what to speak."""
        backend = self._backend
        if entry is None:
            return "no_answer", ""
        if entry.outcome not in (NOT_HEARD, PARTIAL):
            return "answer_heard", ""
        if backend.state != IDLE:
            return "backend_not_idle", ""
        if not entry.run_id or entry.run_id != backend.last_finished_run_id:
            return "other_run", ""
        if backend.epoch != backend.last_finished_epoch:
            return "newer_run", ""
        answer_turn = max(backend.last_finished_turn_ids or (entry.turn_id or 0,))
        if backend.last_task_turn_id > answer_turn:
            return "task_since_answer", ""
        if entry.released_mono is None:
            return "never_released", ""
        if self._ctx.clock.monotonic() - entry.released_mono > self._config.delegation.replay.ttl_s:
            return "ttl_expired", ""
        if entry.replayed:
            return "already_replayed", ""
        text = entry.text if entry.outcome == NOT_HEARD else _unheard_part(entry.text, entry.heard_text)
        whole_below = self._config.delegation.replay.whole_if_heard_below
        if text and whole_below > 0 and len(entry.heard_text) < whole_below * len(entry.text):
            text = entry.text  # F1: almost nothing was heard, so the whole answer is repeated
        if not text:
            return "question_heard", ""
        return "", text

    # -- decisions ------------------------------------------------------------------------------

    def _apply(self, turn: _Turn, text: str, decision: Decision, context: DecisionContext) -> None:
        turn.decided = Stamp.now(self._ctx.clock, self._ctx.audio_now())
        user_entry = self.transcript.add_user(text, turn_id=turn.turn_id, delegated=decision.delegate)
        self._log(
            "delegation_decision",
            turn_id=turn.turn_id,
            delegate=decision.delegate,
            request=decision.request,
            filler=decision.filler_text,
            filler_chars=len(decision.filler_text),
            backend_state=context.state,
            latency_ms=int(decision.latency_ms),
            user_stop_to_decision_ms=_ms(turn.turn_end, turn.decided),
            usage=decision.usage,
            repair=decision.repair,
            text=text,
        )
        # Earlier context (acks, heard fillers, delivery notes) must reach the gateway before the delegation.
        self._flush_context()
        filler = self._guard_filler(turn, decision)
        if decision.delegate:
            if self._backend.unavailable or not self._link_open:
                self._speak_local_apology(turn, reason="backend_unavailable")
                return
            if self._backend.state != WORKING:
                self._backend.current_request = text
            if decision.request == REQUEST_TASK or self._backend.state != WORKING:
                self._backend.last_task_turn_id = turn.turn_id  # the gateway treats IDLE requests as tasks
            self._send(
                "delegate",
                turn_id=turn.turn_id,
                request=decision.request,
                run_input=[{"seq": user_entry.seq, "text": text}],
            )
            if filler and self._config.delegation.filler_dedupe.enabled and decision.repair != LIVENESS_REPLY:
                never_empty = decision.repair in (QUESTION_NEVER_SILENT, PING_AS_STATUS)
                filler = self._dedupe_filler(turn, filler, backend_state=context.state, never_empty=never_empty)
            if filler and self._speaks_delegated_filler:
                self._queue_frontend(turn, decision, filler, kind="filler")
            elif filler:
                reason = "silent_frontend" if self._config.delegation.silent_frontend else "speak_when_delegating"
                self._log("filler_silenced", turn_id=turn.turn_id, text=filler, reason=reason)
        elif filler:
            self._queue_frontend(turn, decision, filler, kind="direct")
        spoken_now = filler and (not decision.delegate or self._speaks_delegated_filler)
        if turn.requested and not spoken_now:
            # A client response.create always gets a response (Realtime semantics): an empty one here.
            self._push(CallsItem(batch_id=f"empty_response_{turn.turn_id}", calls=[]))
        self._wake.set()

    @property
    def _speaks_delegated_filler(self) -> bool:
        """D4 ``speak_when_delegating``, unless the VoiceBench ``silent_frontend`` switch is on."""
        delegation = self._config.delegation
        return delegation.speak_when_delegating and not delegation.silent_frontend

    def _guard_filler(self, turn: _Turn, decision: Decision) -> str:
        """The filler of an F2 decision (R1: a liveness line, R2: an alternative); else the model's."""
        if decision.repair == LIVENESS_REPLY and self._liveness_lines:
            line = self._liveness_lines[self._liveness_next % len(self._liveness_lines)]
            self._liveness_next += 1
            return line  # rotated, never de-duplicated (section 2.3)
        if decision.repair == QUESTION_NEVER_SILENT and self._filler_alternatives:
            for alternative in self._filler_alternatives:
                if _filler_key(alternative) not in self._recent_fillers:
                    self._alternative_used[alternative] = turn.turn_id
                    return alternative
            return self._least_recent_alternative(turn)
        return decision.filler_text

    def _least_recent_alternative(self, turn: _Turn) -> str:
        alternative = min(self._filler_alternatives, key=lambda line: self._alternative_used.get(line, -1))
        self._alternative_used[alternative] = turn.turn_id
        return alternative

    def _dedupe_filler(self, turn: _Turn, filler: str, *, backend_state: str, never_empty: bool = False) -> str:
        """G1: a delegated filler that repeats a recent one is dropped (WORKING) or replaced.

        ``never_empty`` (F2 R2/R3 decisions): replaced in every state, and with every alternative recent the
        least recently used one is spoken instead of nothing.
        """
        key = _filler_key(filler)
        if key not in self._recent_fillers:
            self._recent_fillers.append(key)
            return filler
        if backend_state != WORKING or never_empty:
            for alternative in self._filler_alternatives:
                if _filler_key(alternative) not in self._recent_fillers:
                    self._recent_fillers.append(_filler_key(alternative))
                    self._alternative_used[alternative] = turn.turn_id
                    self._log("filler_deduped", turn_id=turn.turn_id, text=filler, action="replaced", new=alternative)
                    return alternative
            if never_empty and self._filler_alternatives:
                alternative = self._least_recent_alternative(turn)
                self._recent_fillers.append(_filler_key(alternative))
                self._log(
                    "filler_deduped", turn_id=turn.turn_id, text=filler, action="replaced", new=alternative,
                    reason="least_recent",
                )  # fmt: skip
                return alternative
        reason = "working" if backend_state == WORKING else "all_recent"
        self._log("filler_deduped", turn_id=turn.turn_id, text=filler, action="dropped", reason=reason)
        return ""

    def _queue_frontend(self, turn: _Turn, decision: Decision, text: str, *, kind: str) -> None:
        entry = self.transcript.add_spoken("frontend_speech", "frontend", text, turn_id=turn.turn_id)
        self._push(
            SpeechItem(
                kind=kind,
                text=text,
                entry=entry,
                turn_id=turn.turn_id,
                usage=dict(decision.usage),
                on_first_audio=lambda: self._first_audio(turn, "filler"),
            )
        )

    def _speak_local_apology(self, turn: _Turn, *, reason: str) -> None:
        text = "Sorry, I can't reach the system that handles requests right now. Please try again in a moment."
        entry = self.transcript.add_spoken("controller_speech", "controller", text, turn_id=turn.turn_id)
        self._log("apology", turn_id=turn.turn_id, reason=reason)
        self._push(SpeechItem(kind="apology", text=text, entry=entry, turn_id=turn.turn_id))

    def _flush_context(self) -> None:
        entries = self.transcript.take_context()
        if not entries or not self._link_open:
            return
        self._send("history.append", entries=[entry.wire() for entry in entries])
        self._log("history_sync", seqs=[entry.seq for entry in entries])

    # -- backend lane ---------------------------------------------------------------------------

    def _on_backend_message(self, data: dict[str, Any]) -> None:
        if self._closed:
            return
        kind = data["type"]
        handler = getattr(self, f"_gw_{kind.replace('.', '_')}", None)
        if handler is None:
            self._log("gateway_message_ignored", type=kind)
            return
        try:
            handler(data)
        except WireProtocolError:
            raise
        except Exception:  # noqa: BLE001 - one bad gateway message must not end the session
            logger.exception(f"[{self._ctx.session_id}] handling gateway {kind} failed")

    def _gw_session_ready(self, data: dict[str, Any]) -> None:
        self._log("backend_ready", worker_pid=data.get("worker_pid"), **_pick(data, "hermes_version", "model"))
        if self._backend.ready is not None and not self._backend.ready.done():
            self._backend.ready.set_result(data)

    def _gw_session_configured(self, data: dict[str, Any]) -> None:
        if data.get("applied", True):
            self._tools_sha256 = self._pending_tools_sha256
        self._log(
            "backend_configured",
            session_tools_sha256=self._tools_sha256,
            tools_canonical_sha256=self._pending_tools_canonical,
            **_pick(
                data,
                "applied",
                "tools",
                "backend_features",
                "backend_catalog_sha256",
                "backend_soul_sha256",
                "backend_system_sha256",
                "backend_domain",
            ),
        )
        if self._backend.configured is not None and not self._backend.configured.done():
            self._backend.configured.set_result(data)

    def _gw_history_ack(self, data: dict[str, Any]) -> None:
        self._log("history_ack", committed_upto=data["committed_upto"])

    def _gw_state(self, data: dict[str, Any]) -> None:
        backend = self._backend
        previous = backend.state
        backend.state, backend.epoch, backend.run_id = data["state"], int(data["epoch"]), str(data.get("run_id") or "")
        if backend.state == WORKING and previous != WORKING:
            backend.run_started = self._ctx.clock.monotonic()
        if backend.state == WORKING and backend.epoch != self._proactive_epoch:
            # M1: a new run (a pending_steer run follows WORKING directly) gets its own budget and timer.
            self._proactive_epoch, self._proactive_count = backend.epoch, 0
            self._mark_activity()
        if backend.state != WORKING:
            backend.run_started = None
        self._log("backend_state", state=backend.state, epoch=backend.epoch, run_id=backend.run_id, previous=previous)

    def _gw_action(self, data: dict[str, Any]) -> None:
        self._log("backend_action", action=data["kind"], **_pick(data, "turn_id", "request", "run_id", "epoch"))
        if data["kind"] in ("start", "continue"):
            turn = self._turns.get(int(data["turn_id"]))
            if turn is not None:
                self._backend.current_request = turn.text

    def _gw_activity(self, data: dict[str, Any]) -> None:
        preview = str(data.get("preview") or "")[: self._config.frontend.activity_preview_chars]
        line = f"{data['name']}: {data['event'].replace('tool_', '')}" + (f" ({preview})" if preview else "")
        self._backend.activity.append(line)

    def _gw_tool_call(self, data: dict[str, Any]) -> None:
        self._relay.on_call(data)

    def _gw_tool_cancel(self, data: dict[str, Any]) -> None:
        self._relay.on_cancel(str(data["call_id"]), str(data.get("reason") or ""))

    def _gw_answer(self, data: dict[str, Any]) -> None:
        raw = str(data.get("text") or "")
        if data["kind"] == "hermes" and self._config.output.clean_answers:
            cleaned = clean_for_speech(raw)
            if cleaned != " ".join(raw.split()):
                self._log(
                    "answer_cleaned", answer_id=data["answer_id"], chars_before=len(raw), chars_after=len(cleaned)
                )
            raw = cleaned
        text = " ".join(raw.split())
        turn_ids = [int(value) for value in data.get("turn_ids") or () if value is not None]
        turn = self._turns.get(turn_ids[0]) if turn_ids else None
        self._log(
            "backend_answer",
            answer_id=data["answer_id"],
            answer_kind=data["kind"],
            run_id=data.get("run_id"),
            turn_ids=turn_ids,
            chars=len(text),
            reason=data.get("reason"),
            text=text,
        )
        if not text:
            return
        if data["kind"] == "hermes":
            entry = self.transcript.add_spoken(
                "backend_answer",
                "hermes",
                text,
                turn_id=turn.turn_id if turn else None,
                answer_id=str(data["answer_id"]),
                run_id=str(data.get("run_id") or ""),
            )
            kind = "answer"
        else:
            entry = self.transcript.add_spoken(
                "controller_speech", "controller", text, turn_id=turn.turn_id if turn else None
            )
            kind = "apology"
        usage = data.get("usage") or {}
        self._push(
            SpeechItem(
                kind=kind,
                text=text,
                entry=entry,
                turn_id=turn.turn_id if turn else None,
                usage={k: int(usage.get(k, 0) or 0) for k in ("input_tokens", "output_tokens")},
                on_first_audio=(lambda: self._first_audio(turn, "answer")) if turn is not None else None,
            )
        )

    def _gw_status(self, data: dict[str, Any]) -> None:
        task = asyncio.create_task(self._speak_status(data), name="status-verbalize")
        self._side_tasks.add(task)
        task.add_done_callback(self._side_tasks.discard)

    async def _speak_status(self, data: dict[str, Any]) -> None:
        summary = data.get("summary") or {}
        epoch = self._backend.epoch
        verbalized = await self._parts.verbalizer.verbalize(summary)
        turn = self._turns.get(int(data["turn_id"])) if data.get("turn_id") is not None else None
        if self._closed or not verbalized.text:
            return
        if not self._status_current(epoch):
            # The run it describes already finished: "still working" would now be false.
            self._log("status_dropped", turn_id=data.get("turn_id"), reason="run_finished", text=verbalized.text)
            return
        self._log("status_spoken", turn_id=data.get("turn_id"), source=verbalized.source, text=verbalized.text)
        entry = self.transcript.add_spoken(
            "status_speech", "frontend", verbalized.text, turn_id=turn.turn_id if turn else None
        )
        self._push(
            SpeechItem(
                kind="status",
                text=verbalized.text,
                entry=entry,
                turn_id=turn.turn_id if turn else None,
                usage=verbalized.usage,
                meta={"epoch": epoch},
            )
        )

    def _gw_backend_run_done(self, data: dict[str, Any]) -> None:
        backend = self._backend
        turn_ids = tuple(int(value) for value in data.get("turn_ids") or () if value is not None)
        first = self._turns.get(turn_ids[0]) if turn_ids else None
        backend.last_finished_run_id = str(data.get("run_id") or "")
        backend.last_finished_epoch = int(data.get("epoch") or 0)
        backend.last_finished_turn_ids = turn_ids
        backend.last_finished_request = first.text if first is not None else backend.current_request
        self._log(
            "backend_run_done",
            **_pick(data, "run_id", "epoch", "status", "history", "turn_ids", "usage", "tool_call_ids"),
        )

    def _gw_worker(self, data: dict[str, Any]) -> None:
        self._log("worker", **_pick(data, "event", "pid", "reason", "start_ms", "rss_mb"))

    def _gw_error(self, data: dict[str, Any]) -> None:
        code, text, fatal = str(data["code"]), str(data["message"]), bool(data["fatal"])
        self._log("backend_error", code=code, message=text, fatal=fatal)
        logger.warning(f"[{self._ctx.session_id}] backend error {code}: {text}")
        error = _GatewayError(code, text)
        for future in (self._backend.ready, self._backend.configured):
            if future is not None and not future.done():
                future.set_exception(error)
                future.exception()  # mark retrieved: the waiter may already be gone
        if fatal:
            self._backend.unavailable = f"{code}: {text}"
            self._ctx.emit(
                ev.error(f"backend unavailable: {text}", code="backend_unavailable", error_type="server_error")
            )

    # -- output lane ----------------------------------------------------------------------------

    def _push(self, item: SpeechItem | CallsItem) -> None:
        self._queue.push(item)
        self._wake.set()

    async def _output_loop(self) -> None:
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            while self._running is None and not self._closed:
                item, dropped = self._queue.next(
                    user_speaking=self._awaiting > 0,
                    deciding=self._deciding(),
                    current_turn=self._decision_turn.turn_id if self._decision_turn else None,
                )
                self._record_dropped(dropped)
                if item is None:
                    break
                if (
                    isinstance(item, SpeechItem)
                    and item.kind == "status"
                    and not self._status_current(item.meta["epoch"])
                ):
                    self._log("status_dropped", turn_id=item.turn_id, reason="run_finished", text=item.text)
                    self._settle(item.entry, NOT_HEARD, reason="superseded")
                    continue
                await self._send_output(item)
            # A context entry may have become final while speaking.
            self._flush_context()

    # -- M1: proactive status ---------------------------------------------------------------------

    def _mark_activity(self) -> None:
        """Audio was released or the user spoke: the silence timer starts again."""
        self._quiet_since = self._ctx.clock.monotonic()

    async def _proactive_loop(self) -> None:
        after_s = self._config.output.proactive_status.after_s
        poll = min(0.5, max(0.02, after_s / 4))
        while not self._closed:
            await asyncio.sleep(poll)
            self._maybe_proactive_status()

    def _maybe_proactive_status(self) -> None:
        """One short status line after ``after_s`` of silence while WORKING (no LLM call)."""
        settings = self._config.output.proactive_status
        backend = self._backend
        if backend.state != WORKING or backend.unavailable or self._proactive_count >= settings.max_per_run:
            return
        now = self._ctx.clock.monotonic()
        busy = (
            self._awaiting > 0
            or self._deciding()
            or self._running is not None
            or len(self._queue) > 0
            or self.progress.active
        )
        if busy or self._quiet_since is None:
            self._quiet_since = now  # silence counts from the end of speech, either side
            return
        silence = now - self._quiet_since
        if silence < settings.after_s or not self._proactive_lines:
            return
        text = self._proactive_lines[self._proactive_next % len(self._proactive_lines)]
        self._proactive_next += 1
        self._proactive_count += 1
        self._log(
            "status_proactive",
            turn_id=self._turn_counter or None,
            run_id=backend.run_id,
            epoch=backend.epoch,
            text=text,
            silence_wall_s=round(silence, 2),
        )
        if self._config.delegation.silent_frontend:
            # Counted against max_per_run above, so this is logged at most that often per run.
            self._log("proactive_status_silenced", run_id=backend.run_id, epoch=backend.epoch, text=text)
            return
        entry = self.transcript.add_spoken("status_speech", "frontend", text)
        self._push(SpeechItem(kind="status", text=text, entry=entry, meta={"epoch": backend.epoch, "proactive": True}))

    def _status_current(self, epoch: int) -> bool:
        """Whether the run a status report described is still the one running."""
        return self._backend.state == WORKING and self._backend.epoch == epoch

    def _record_dropped(self, dropped: list[Dropped]) -> None:
        for drop in dropped:
            self._log("output_dropped_stale", item_kind=drop.item.kind, turn_id=drop.item.turn_id, text=drop.item.text)
            self._settle(drop.item.entry, NOT_HEARD, reason=drop.reason)

    async def _send_output(self, item: SpeechItem | CallsItem) -> None:
        settings = self._ctx.settings()
        writer = ResponseWriter(self._ctx.emit, self._ctx.conversation, modality=settings.output_modality)
        running = _Running(writer=writer, item=item)
        self._running = running
        writer.open()
        self.progress.generating = True
        try:
            if isinstance(item, CallsItem):
                for call_id, name, arguments in item.calls:
                    writer.function_call(call_id=call_id, name=name, arguments=arguments)
                writer.finish("completed", usage=ev.usage_object())
                return
            now = self._ctx.clock.monotonic()
            waited_ms = int((now - item.queued_mono) * 1000)
            self._log("output_release", item_kind=item.kind, turn_id=item.turn_id, waited_ms=waited_ms)
            if item.entry is not None and item.entry.released_mono is None:
                item.entry.released_mono = now
            self._mark_activity()
            running.task = asyncio.current_task()
            await self._speak(running, item, settings)
        finally:
            self.progress.generating = False
            if self._running is running:
                self._running = None
            self._log("response_done", response_id=writer.response_id, status=writer.status, item_kind=_kind(item))

    async def _speak(self, running: _Running, item: SpeechItem, settings: SessionSettings) -> None:
        writer = running.writer
        prepared = self._ctx.output.prepare(
            item.text,
            kind="answer" if item.kind in ("answer", "replay", "status", "apology") else "filler",
            modality=settings.output_modality,
            voice=self._ctx.voice(),
            out_format=settings.output_format,
        )
        message_item = writer.begin_message(kind=item.kind)
        running.item_id = message_item.item_id
        if item.entry is not None:
            self._unheard[message_item.item_id] = item.entry
        running.speak_task = asyncio.create_task(
            self._ctx.output.speak(
                prepared,
                message_item,
                self.progress,
                out_format=settings.output_format,
                on_first_audio=item.on_first_audio,
            )
        )
        status = "completed"
        try:
            await running.speak_task
        except asyncio.CancelledError:
            if not running.interrupted:
                raise
            status = "incomplete"
        except Exception as exc:  # noqa: BLE001 - a TTS failure fails this response, not the session
            logger.exception(f"[{self._ctx.session_id}] speaking {item.kind} failed")
            self._ctx.emit(ev.error(f"response failed: {exc}", code="response_failed", error_type="server_error"))
            status = "failed"
        finally:
            await prepared.aclose()
        if running.interrupted:
            return  # _interrupt_output closed the response and recorded the outcome
        message_item.close(status="completed" if status == "completed" else "incomplete")
        usage = ev.usage_object(**item.usage) if self._voice.protocol.emit_usage and item.usage else ev.usage_object()
        writer.finish(
            "completed" if status == "completed" else "failed",
            usage=usage,
            reason="server_error" if status == "failed" else None,
        )
        if settings.output_modality != "audio" and item.entry is not None:
            self._unheard.pop(message_item.item_id, None)
            self._settle(item.entry, HEARD)
        elif status == "failed" and item.entry is not None:
            self._unheard.pop(message_item.item_id, None)
            heard = self.progress.heard_text(message_item.item_id)
            self._settle(item.entry, PARTIAL if heard else NOT_HEARD, heard_text=heard, reason="tts_failed")

    async def _interrupt_output(self, *, reason: str) -> None:
        """Barge-in: stop the running response, record what was heard, drop unplayed audio."""
        running = self._running
        heard_ms = self.progress.heard_ms
        if running is not None and isinstance(running.item, SpeechItem) and not running.writer.done:
            running.interrupted = True
            if running.speak_task is not None and not running.speak_task.done():
                running.speak_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await running.speak_task
            running.writer.finish("cancelled", reason=reason)
        for item_id, entry in list(self._unheard.items()):
            span = self.progress.item(item_id)
            heard = self.progress.heard_text(item_id)
            del self._unheard[item_id]
            if span is not None and span.end_ms <= heard_ms and heard == " ".join(entry.text.split()):
                self._settle(entry, HEARD)
            else:
                self._settle(entry, PARTIAL if heard else NOT_HEARD, heard_text=heard, reason=reason)
        self._log("barge_in", reason=reason, heard_ms=int(heard_ms), sent_ms=int(self.progress.sent_ms))
        self.progress.discard_unplayed()

    def _settle(self, entry: Entry | None, outcome: str, *, heard_text: str = "", reason: str = "") -> None:
        if entry is None or entry.final:
            return
        note = self.transcript.settle(entry, outcome, heard_text=heard_text, reason=reason)
        self._log(
            "playback_outcome",
            seq=entry.seq,
            entry_kind=entry.kind,
            origin=entry.origin,
            outcome=entry.outcome,
            heard_chars=len(entry.heard_text)
            if entry.outcome == PARTIAL
            else (len(entry.text) if entry.outcome == HEARD else 0),
            reason=reason,
            answer_id=entry.answer_id,
            heard_text=entry.heard_text[:300] if entry.outcome == PARTIAL else "",
        )
        if note is not None:
            self._log("delivery_note", seq=note.seq, answer_id=note.answer_id, outcome=note.outcome)
        self._flush_context()

    # -- helpers --------------------------------------------------------------------------------

    def _first_audio(self, turn: _Turn | None, kind: str) -> None:
        if turn is None:
            return
        now = Stamp.now(self._ctx.clock, self._ctx.audio_now())
        if kind == "filler" and not turn.first_audio_logged:
            turn.first_audio_logged = True
            self._log(
                "filler_timing",
                turn_id=turn.turn_id,
                user_stop_to_first_audio_ms=_ms(turn.turn_end, now),
                decision_to_first_audio_ms=_ms(turn.decided, now),
            )
        if kind == "answer" and not turn.answer_audio_logged:
            turn.answer_audio_logged = True
            breakdown = {
                "turn_id": turn.turn_id,
                "user_stop_to_asr_final_ms": _ms(turn.turn_end, turn.asr_final),
                "user_stop_to_decision_ms": _ms(turn.turn_end, turn.decided),
                "user_stop_to_first_audio_ms": _ms(turn.turn_end, now),
            }
            logger.info(f"[{self._ctx.session_id}] answer latency {breakdown}")
            self._log("turn_latency", **breakdown)

    def _send(self, type_: str, **fields: Any) -> None:
        self._parts.link.send(message(VOICE_TO_GATEWAY, type_, **fields))

    def _send_tool_result(self, call_id: str, epoch: int, output: str, local: bool) -> None:
        if self._link_open and not self._closed:
            self._send("tool.result", call_id=call_id, epoch=epoch, output=output, local=local)

    def _log(self, event: str, /, **data: Any) -> None:
        self._ctx.event_log.write(event, self._ctx.session_id, {"audio_ms": int(self._ctx.audio_now()), **data})


class _GatewayError(Exception):
    """A gateway ``error`` answering a pending open/configure."""

    def __init__(self, code: str, text: str) -> None:
        super().__init__(f"{code}: {text}")
        self.code = code
        self.text = text


def _ms(start: Stamp | None, end: Stamp | None) -> int | None:
    return None if start is None or end is None else int(round((end.mono - start.mono) * 1000))


def _pick(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data}


def _kind(item: SpeechItem | CallsItem) -> str:
    return "function_calls" if isinstance(item, CallsItem) else item.kind


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _unheard_part(text: str, heard_text: str) -> str:
    """A partly heard answer from the start of the sentence that was cut ("" if its last sentence was heard)."""
    heard = " ".join(heard_text.split())
    if len(heard.rstrip(".!?")) >= len(text.rstrip(".!?")):
        return ""
    start = 0
    for part in _SENTENCE_END.split(text):
        end = start + len(part)
        if end > len(heard.rstrip()):
            return text[start:].strip()
        start = end + 1
    return ""


def _filler_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


__all__ = ["DelegationTurnManager", "SessionParts"]
