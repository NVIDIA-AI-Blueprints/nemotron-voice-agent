# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AgentPort`` over the text prototype's ``FrontendBackendAgent``.

One runner per Realtime session. It owns the session's immutable
``SessionState`` and swaps it only when a call returns, so cancelling an
in-flight ``respond()`` (barge-in while thinking) leaves the conversation
exactly as it was. The agent itself is assembled per session with
``assemble_agent`` from the shared LLM clients, because each session has its
own tools and instructions.

With ``normalization`` on, the runner is also where both hooks run: the
transcript hook on the text of each agent turn, and the tool-argument hook on
every batch of backend tool calls before it reaches the client. Calls answered
locally never leave the runner; their results are fed back to the agent at once
(all calls local) or merged with the client's outputs on ``resume``.

With ``barge_in.while_thinking: frontend_verdict`` the runner also serves the
barge-in review: it records the running turn's delegation (``in_flight``), asks
the frontend about speech during that turn (``probe``) without touching the
state, carries a probe out on the state it was made on (``proceed``), and can
hold (stage) the running turn's text result until the verdict. Staging is bound
to the one call in flight when it began; every new call starts with it off.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from examples.frontend_backend_verdict.text.agent import FrontendBackendAgent, FrontendStep, assemble_agent
from examples.frontend_backend_verdict.text.backend_context import backend_system_prompt
from examples.frontend_backend_verdict.text.config import Config
from examples.frontend_backend_verdict.text.delegation import TASK_CONTINUE
from examples.frontend_backend_verdict.text.events import EventSink, InternalEvent
from examples.frontend_backend_verdict.text.frontend import Delegate, DirectAnswer
from examples.frontend_backend_verdict.text.llm import ChatClient
from examples.frontend_backend_verdict.text.messages import (
    CONTINUES_TURN,
    AgentTurn,
    Message,
    RoleTotals,
    ToolCall,
    ToolResult,
    UsageTotals,
)
from examples.frontend_backend_verdict.text.prompts import load_catalog, render
from examples.frontend_backend_verdict.text.session import SessionState
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.agent.history_repair import repair_interrupted_answer
from examples.frontend_backend_verdict.voice.agent.instructions import ResolvedInstructions, session_agent_config
from examples.frontend_backend_verdict.voice.agent.port import (
    VERDICT_CONTINUE,
    VERDICT_NEW,
    AgentReply,
    InFlight,
    Probe,
    ReplyUsage,
    RoleUsage,
)
from examples.frontend_backend_verdict.voice.agent.tools import realtime_tools_to_specs, to_outgoing, to_results
from examples.frontend_backend_verdict.voice.agent.write_gate import (
    ConfirmationProposal,
    GateState,
    WriteGate,
    WriteGateSettings,
    resolve_gate,
    spoken_presentation,
)
from examples.frontend_backend_verdict.voice.config import InstructionsConfig, ToolsConfig, VoiceConfig
from examples.frontend_backend_verdict.voice.errors import HistoryRepairError, ProbeStateError, VoiceConfigError
from examples.frontend_backend_verdict.voice.normalization import (
    ARGUMENT_NORMALIZED,
    CALL_ANSWERED_LOCALLY,
    LOCAL_ROUNDS_EXHAUSTED,
    TRANSCRIPT_NORMALIZED,
    NormalizationSettings,
)
from examples.frontend_backend_verdict.voice.normalization.arguments import (
    ArgumentNormalizer,
    FailureKey,
    Screening,
)
from examples.frontend_backend_verdict.voice.normalization.prompts import with_frontend_note
from examples.frontend_backend_verdict.voice.normalization.transcript import TranscriptNormalizer


@dataclass(frozen=True, slots=True)
class FrontendVerdictSettings:
    """What the runner needs for ``barge_in.while_thinking: frontend_verdict``.

    The in-progress note itself is a prompt template (``barge_in.frontend_verdict.note_key``),
    included by the frontend template when ``in_progress`` is set.
    """

    same_query_guard: bool = True


def frontend_verdict_settings(config: VoiceConfig) -> FrontendVerdictSettings | None:
    """The runner's settings for the frontend verdict, or ``None`` when it is off."""
    if not config.barge_in.uses_frontend_verdict:
        return None
    return FrontendVerdictSettings(same_query_guard=config.barge_in.frontend_verdict.same_query_guard)


def _normalized_request(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def same_request(a: str, b: str) -> bool:
    """Whether two delegation queries are equal after case-folding and removing punctuation and spacing."""
    return _normalized_request(a) == _normalized_request(b)


@dataclass(frozen=True, slots=True)
class AgentClients:
    """LLM clients shared by every session (the OpenAI client is safe to share)."""

    backend: ChatClient
    frontend: ChatClient | None = None


def _role_usage(totals: RoleTotals) -> RoleUsage:
    usage = totals.usage
    return RoleUsage(
        calls=totals.calls,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        total_tokens=usage.total_tokens or usage.prompt_tokens + usage.completion_tokens,
        latency_ms=totals.latency_ms,
    )


def _reply(turn: AgentTurn, totals: UsageTotals | None = None, calls: tuple[ToolCall, ...] | None = None) -> AgentReply:
    """The port reply for ``turn``; ``totals`` sums several agent steps, ``calls`` replaces its calls."""
    totals = totals or turn.usage
    usage = totals.usage
    reply_usage = ReplyUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_tokens=usage.cached_tokens,
        frontend=_role_usage(totals.frontend),
        backend=_role_usage(totals.backend),
    )
    outgoing = turn.tool_calls if calls is None else calls
    if outgoing:
        return AgentReply(calls=to_outgoing(outgoing), usage=reply_usage)
    return AgentReply(text=turn.final_text or "", usage=reply_usage)


def _rewrite_pending_calls(state: SessionState, calls: Sequence[ToolCall]) -> SessionState:
    """Store canonical arguments in the suspended backend group (paired and ``backend_only`` alike)."""
    pending = state.pending
    if pending is None or not pending.backend_history.messages:
        return state
    by_id = {call.id: call for call in calls}
    messages = list(pending.backend_history.messages)
    last = messages[-1]
    rewritten = tuple(by_id.get(call.id, call) for call in last.tool_calls)
    if rewritten == last.tool_calls:
        return state
    messages[-1] = replace(last, tool_calls=rewritten)
    history = replace(pending.backend_history, messages=tuple(messages))
    return replace(state, pending=replace(pending, backend_history=history))


@dataclass(frozen=True, slots=True)
class _Settled:
    """One ``respond``/``resume`` outcome, committed only when the call completes."""

    reply: AgentReply
    state: SessionState
    held: dict[str, str]
    keys: dict[str, FailureKey]
    gate: GateState = GateState()


class TextAgentRunner:
    """One session's agent: the text prototype plus the session's state."""

    def __init__(
        self,
        *,
        base_config: Config,
        tools_config: ToolsConfig,
        instructions_config: InstructionsConfig,
        clients: AgentClients,
        sink: EventSink,
        session_id: str,
        seed_greeting: str = "",
        normalization: NormalizationSettings | None = None,
        barge_in: FrontendVerdictSettings | None = None,
        prompt_context: Mapping[str, Any] | None = None,
        write_gate: WriteGateSettings | None = None,
    ) -> None:
        """Build the initial agent (no client tools, no instructions yet).

        ``prompt_context`` holds the Jinja variables of the prompt templates
        (``config.prompt_context``); the frontend also gets ``in_progress``.
        """
        self._base = base_config
        self._barge_in = barge_in
        self._prompt_context = dict(prompt_context or {})
        self._tools_config = tools_config
        self._instructions_config = instructions_config
        self._clients = clients
        self._sink = sink
        self._state = SessionState(session_id=session_id)
        self._normalization = normalization or NormalizationSettings()
        transcript = self._normalization.transcript
        self._transcript = TranscriptNormalizer(transcript) if transcript.enabled else None
        self._arguments = self._argument_normalizer()
        #: Retry-guard keys of calls that failed permanently in this session.
        self._failed: frozenset[FailureKey] = frozenset()
        #: Local results of the outstanding calls that were not sent, merged in on ``resume``.
        self._held: dict[str, str] = {}
        #: Retry-guard keys of the outstanding calls that were sent.
        self._sent_keys: dict[str, FailureKey] = {}
        #: The running turn's delegation (frontend verdict note), set between delegation and the call's end.
        self._in_flight: InFlight | None = None
        #: Serial number of the respond/resume/proceed call in flight (``None`` when idle).
        self._active_call: int | None = None
        self._calls = 0
        #: The call whose text result is held instead of committed (``begin_staging``).
        self._staging_call: int | None = None
        #: The held result and its retry-guard keys, until ``end_staging``.
        self._staged: tuple[_Settled, frozenset[FailureKey]] | None = None
        self._agent: FrontendBackendAgent | None = None
        #: ``write_gate``: settings, the session's resolved gate, its state, and which summaries were heard.
        self._gate_settings = write_gate or WriteGateSettings()
        self._gate: WriteGate | None = None
        self._gate_state = GateState()
        self._heard: dict[str, bool] = {}
        self.config: Config = base_config
        self.resolved_instructions: ResolvedInstructions | None = None
        self.tool_specs: tuple[ToolSpec, ...] = ()
        self.configure(tools=(), instructions="")
        if seed_greeting:
            self.seed_assistant(seed_greeting)

    # -- AgentPort ------------------------------------------------------------

    @property
    def backend_only(self) -> bool:
        """Whether the frontend is disabled."""
        return not self._base.frontend_enabled

    @property
    def state(self) -> SessionState:
        """The current (immutable) session state."""
        return self._state

    def configure(self, *, tools: Sequence[Mapping[str, Any]], instructions: str) -> None:
        """Rebuild the agent for new tools/instructions; the conversation state is kept."""
        if self._tools_config.source == "client":
            specs = realtime_tools_to_specs(tools)
        else:
            specs = self._tools_config.config_tool_specs
        config, resolved = session_agent_config(
            self._base, self._instructions_config, session_instructions=instructions, tools=specs
        )
        if self._transcript is not None:
            config = with_frontend_note(config, self._normalization.transcript.frontend_note_key)
        self._agent = assemble_agent(
            config,
            tools=specs,
            event_sink=self._sink,
            frontend_client=self._clients.frontend if config.frontend_enabled else None,
            backend_client=self._clients.backend,
            prompt_context=self._prompt_context,
        )
        resolved_gate = resolve_gate(self._gate_settings, specs)
        if resolved_gate is not None and not config.frontend_enabled:
            raise VoiceConfigError("write_gate needs the frontend: it judges the caller's confirmation")
        self._gate = WriteGate(resolved_gate, self._emit) if resolved_gate is not None else None
        self.config = config
        self.resolved_instructions = resolved
        self.tool_specs = tuple(specs)

    async def respond(self, text: str) -> AgentReply:
        """Run one user turn; the state is replaced only if the call completes."""
        assert self._agent is not None  # noqa: S101 - built in __init__
        call = self._begin_call()
        try:
            text = self._normalize_transcript(text)
            gate, bound = self._gate_turn()
            if not self._base.frontend_enabled:
                turn, state = await self._agent.send(text, self._state)
            else:
                confirmation = {"summary": bound.summary_text} if bound is not None else None
                step = await self._agent.decide_turn(text, self._state, confirmation=confirmation)
                gate, step = self._gate_decision(gate, bound, step)
                turn, state = await self._continue(step)
            settled = await self._settle(turn, state, self._failed, gate)
            return self._commit_or_stage(settled, self._failed, call)
        finally:
            self._end_call(call)

    async def resume(
        self,
        outputs: Mapping[str, str],
        *,
        user_text: str | None = None,
        user_meta: Mapping[str, Any] | None = None,
    ) -> AgentReply:
        """Continue a suspended turn with the client's outputs (plus any held local results).

        ``user_text`` (caller speech while the calls were out) follows the tool
        messages as one user message; ``user_meta`` stays on that message only.
        """
        assert self._agent is not None  # noqa: S101 - built in __init__
        call = self._begin_call()
        try:
            failed = self._failed
            if self._arguments is not None:
                failed = failed | {
                    self._sent_keys[call_id]
                    for call_id, output in outputs.items()
                    if call_id in self._sent_keys and self._arguments.is_permanent_failure(output)
                }
            message = None
            if user_text is not None:
                meta = {**(user_meta or {}), CONTINUES_TURN: True}
                message = Message.user(self._normalize_transcript(user_text), meta=meta)
            turn, state = await self._agent.send_tool_results(
                to_results({**self._held, **outputs}), self._state, user_message=message
            )
            settled = await self._settle(turn, state, failed, self._gate_state)
            self._commit(settled, failed)
            return settled.reply
        finally:
            self._end_call(call)

    # -- barge-in frontend verdict ----------------------------------------------

    @property
    def in_flight(self) -> InFlight | None:
        """The running turn's delegation, once the frontend has delegated."""
        return self._in_flight

    async def probe(self, text: str, *, new_words: str, filler_spoken: bool) -> Probe:
        """Ask the frontend about speech during the running turn; the state is not changed."""
        in_flight = self._in_flight
        if self._barge_in is None or in_flight is None:
            raise ProbeStateError("probe() needs barge_in.while_thinking: frontend_verdict and a delegated turn")
        return await self.probe_request(text, in_flight=in_flight, new_words=new_words, filler_spoken=filler_spoken)

    async def probe_request(self, text: str, *, in_flight: InFlight, new_words: str, filler_spoken: bool) -> Probe:
        """The probe for ``in_flight`` (``probe`` uses the running turn's; the verdict replay passes its own)."""
        assert self._agent is not None  # noqa: S101 - built in __init__
        text = self._normalize_transcript(text)
        in_progress = {
            "query": in_flight.query,
            "filler": in_flight.filler_text,
            "filler_heard": filler_spoken,
            "new_words": new_words.strip(),
        }
        started = time.monotonic()
        step = await self._agent.decide_turn(text, self._state, in_progress=in_progress)
        latency_ms = (time.monotonic() - started) * 1000.0
        decision = step.decision
        probe_query = model_task = ""
        guard = self._barge_in is None or self._barge_in.same_query_guard
        if isinstance(decision, Delegate):
            probe_query, model_task = decision.query, decision.task
            if guard and same_request(decision.query, in_flight.query):
                # Restarting the identical request is never useful, whatever the model's task says.
                verdict = VERDICT_CONTINUE
                reason = "model" if model_task == TASK_CONTINUE else "same_query"
            elif model_task == TASK_CONTINUE:
                verdict, reason = VERDICT_CONTINUE, "model"
            else:
                verdict, reason = VERDICT_NEW, ("model" if model_task else "task_invalid")
        elif isinstance(decision, DirectAnswer):
            verdict, reason = VERDICT_NEW, "direct_answer"
        else:
            verdict, reason = VERDICT_NEW, "contract_fallback"
        return Probe(
            verdict=verdict,
            reason=reason,
            running_query=in_flight.query,
            probe_query=probe_query,
            step=step,
            latency_ms=latency_ms,
            frontend=_role_usage(step.totals.frontend),
            model_task=model_task,
        )

    async def proceed(self, probe: Probe) -> AgentReply:
        """Carry out ``probe`` on the state it was made on (raises ``ProbeStateError`` if that changed)."""
        step = probe.step
        if not isinstance(step, FrontendStep) or self._state is not step.session:
            raise ProbeStateError("the conversation state changed since the probe; it cannot be carried out")
        call = self._begin_call()
        try:
            # The probe was decided without a pending summary, so it cannot confirm one.
            gate, bound = self._gate_turn()
            gate, step = self._gate_decision(gate, bound, step)
            turn, state = await self._continue(step)
            settled = await self._settle(turn, state, self._failed, gate)
            return self._commit_or_stage(settled, self._failed, call)
        finally:
            self._end_call(call)

    def begin_staging(self) -> None:
        """Hold a text result of the call in flight instead of committing it."""
        if self._active_call is not None:
            self._staging_call = self._active_call

    def end_staging(self, *, commit: bool) -> None:
        """Stop staging; commit (``commit``) or drop a held result."""
        staged, self._staged = self._staged, None
        self._staging_call = None
        if self._active_call is None:
            self._in_flight = None
        if staged is not None and commit:
            self._commit(*staged)

    def presentation_heard(self, proposal_id: str, *, complete: bool) -> None:
        """Record whether a spoken action summary played to its end before the caller's next turn."""
        self._heard[proposal_id] = complete
        self._emit("write_presentation_heard", proposal_id=proposal_id, complete=complete)

    def repair_last_answer(self, full_text: str, replacement: str) -> None:
        """Rewrite every stored copy of the interrupted answer."""
        self._state = repair_interrupted_answer(
            self._state,
            full_text=full_text,
            replacement=replacement,
            frontend_enabled=self._base.frontend_enabled,
            backend_history=self._base.backend.conversation_history.keeps_backend_turns,
        )

    def seed_assistant(self, text: str) -> None:
        """Append assistant speech the agent did not produce (a greeting) to the owning history."""
        message = Message.assistant(text)
        if self._base.frontend_enabled:
            self._state = replace(self._state, frontend_history=self._state.frontend_history.append(message))
        else:
            self._state = replace(self._state, backend_history=self._state.backend_history.append(message))

    # -- normalization ---------------------------------------------------------

    def _argument_normalizer(self) -> ArgumentNormalizer | None:
        settings = self._normalization.tool_arguments
        if not settings.enabled:
            return None
        catalog = load_catalog(self._base.prompts_path, self._base.prompts.inline)
        guard = settings.retry_guard
        return ArgumentNormalizer(
            settings,
            transcript=self._normalization.transcript,
            invalid_template=catalog.get(settings.invalid_message_key),
            already_failed_template=catalog.get(guard.message_key) if guard.enabled else "",
        )

    def _emit(self, kind: str, **data: Any) -> None:
        self._sink.emit(InternalEvent(kind, self._state.session_id, data))

    def _normalize_transcript(self, text: str) -> str:
        if self._transcript is None:
            return text
        normalized = self._transcript.normalize(text)
        if normalized.changed:
            spans = [{"spoken": span.spoken, "written": span.written} for span in normalized.spans]
            self._emit(TRANSCRIPT_NORMALIZED, raw=text, text=normalized.text, spans=spans)
        return normalized.text

    def _begin_call(self) -> int:
        self._calls += 1
        self._active_call = self._calls
        self._staging_call = None
        self._staged = None
        self._in_flight = None
        return self._calls

    def _end_call(self, call: int) -> None:
        if self._active_call == call:
            self._active_call = None
            if self._staged is None:
                # A staged turn is still the request in progress: keep it for further probes.
                self._in_flight = None
        if self._staging_call == call:
            self._staging_call = None

    async def _continue(self, step: FrontendStep) -> tuple[AgentTurn, SessionState]:
        assert self._agent is not None  # noqa: S101 - built in __init__
        if isinstance(step.decision, Delegate):
            self._in_flight = InFlight(query=step.decision.query, filler_text=step.decision.filler_text)
        return await self._agent.continue_turn(step)

    def _commit_or_stage(self, settled: _Settled, failed: frozenset[FailureKey], call: int) -> AgentReply:
        """Commit, or hold a text result while this call is staging (tool calls always commit)."""
        if self._staging_call == call and settled.reply.text is not None:
            self._staged = (settled, failed)
            return replace(settled.reply, staged=True)
        self._commit(settled, failed)
        return settled.reply

    def _commit(self, settled: _Settled, failed: frozenset[FailureKey]) -> None:
        self._state = settled.state
        self._held = settled.held
        self._sent_keys = settled.keys
        self._failed = failed
        self._gate_state = settled.gate

    # -- write gate -------------------------------------------------------------

    def _gate_turn(self) -> tuple[GateState, ConfirmationProposal | None]:
        """A caller turn starts: the proposal it may confirm (the gate state is committed with the turn)."""
        if self._gate is None:
            return self._gate_state, None
        return self._gate.begin_caller_turn(self._gate_state, self._heard)

    def _gate_decision(
        self, gate: GateState, bound: ConfirmationProposal | None, step: FrontendStep
    ) -> tuple[GateState, FrontendStep]:
        """Judge the bound turn (fail closed) and put the gate's notes into the backend request."""
        if self._gate is None:
            return gate, step
        decision = step.decision
        if bound is not None:
            verdict = decision.confirmation if isinstance(decision, Delegate) else ""
            gate = self._gate.judge(gate, bound, verdict or "unclear")
        if not isinstance(decision, Delegate) or not gate.notes:
            return gate, step
        gate, notes = self._gate.take_notes(gate)
        query = "\n\n".join([decision.query, *notes])
        return gate, replace(step, decision=replace(decision, query=query))

    def _rewrite_answer(self, state: SessionState, text: str, spoken: str) -> SessionState:
        """Store what is actually spoken (framing + summary + question) as the turn's answer."""
        try:
            return repair_interrupted_answer(
                state,
                full_text=text,
                replacement=spoken,
                frontend_enabled=self._base.frontend_enabled,
                backend_history=self._base.backend.conversation_history.keeps_backend_turns,
            )
        except HistoryRepairError as exc:
            self._emit("write_gate_error", error=f"history rewrite skipped: {exc}")
            return state

    def _presented(self, turn: AgentTurn, totals: UsageTotals, state: SessionState, gate: GateState) -> _Settled:
        """The turn's text reply; when a held call awaits its summary, the summary and question follow it."""
        reply = _reply(turn, totals)
        proposal = self._gate.unpresented(gate) if self._gate is not None else None
        if proposal is None or reply.text is None:
            return _Settled(reply, state, {}, {}, gate)
        spoken = spoken_presentation(reply.text, proposal)
        state = self._rewrite_answer(state, reply.text, spoken)
        gate = self._gate.present(gate, proposal)
        return _Settled(replace(reply, text=spoken, presentation=proposal.proposal_id), state, {}, {}, gate)

    # -- settling ---------------------------------------------------------------

    async def _settle(
        self, turn: AgentTurn, state: SessionState, failed: frozenset[FailureKey], gate: GateState
    ) -> _Settled:
        """Screen the turn's tool calls, then gate them, before anything is sent.

        All-local batches (argument normalization, write-gate holds) are answered at
        once and the backend continues, at most ``max_local_rounds`` times for the
        normalizer's own answers. The write gate runs on every path, including the
        normalizer's exhausted fallback: a gated call is never sent without a match
        to a confirmed proposal.
        """
        assert self._agent is not None  # noqa: S101 - built in __init__
        totals = turn.usage
        rounds = gate_rounds = 0
        while turn.tool_calls:
            candidates = turn.tool_calls
            local: dict[str, str] = {}
            keys: dict[str, FailureKey] = {}
            if self._arguments is not None:
                screening = self._arguments.screen(turn.tool_calls, failed)
                state = _rewrite_pending_calls(state, screening.calls)
                self._emit_rewrites(screening)
                keys = dict(screening.keys)
                if not screening.sent and rounds >= self._arguments.settings.max_local_rounds:
                    # Only the local interception is bypassed: the canonical calls go out (if the gate allows).
                    self._emit(LOCAL_ROUNDS_EXHAUSTED, tools=[call.name for call in screening.calls], rounds=rounds)
                    candidates = screening.calls
                else:
                    for answer in screening.local:
                        self._emit(
                            CALL_ANSWERED_LOCALLY,
                            call_id=answer.call_id,
                            tool=answer.tool,
                            argument=answer.argument,
                            value=answer.value,
                            reason=answer.reason,
                            local_round=rounds + 1,
                        )
                    local = {answer.call_id: answer.message for answer in screening.local}
                    candidates = screening.sent
            sent, held = candidates, dict(local)
            if self._gate is not None:
                screened = self._gate.screen(candidates, gate)
                gate, sent = screened.state, screened.allowed
                held.update(screened.held)
                if screened.held and not sent:
                    gate_rounds += 1
                stop = screened.stop or gate_rounds > self._gate.settings.max_unconfirmed_reissues + 1
                proposal = self._gate.unpresented(gate) or gate.proposal(gate.presented)
                if stop and proposal is not None and not sent:
                    # Stop the backend loop for this caller turn and ask again; the call is never sent.
                    results = tuple(ToolResult(tool_call_id=k, content=v) for k, v in held.items())
                    turn, state = self._agent.close_turn(results, spoken_presentation("", proposal), state)
                    gate = self._gate.present(gate, proposal)
                    reply = replace(_reply(turn, totals), presentation=proposal.proposal_id)
                    return _Settled(reply, state, {}, {}, gate)
            if sent:
                keys = {call_id: key for call_id, key in keys.items() if call_id not in held}
                return _Settled(_reply(turn, totals, sent), state, held, keys, gate)
            if local:
                rounds += 1
            results = tuple(ToolResult(tool_call_id=call_id, content=message) for call_id, message in held.items())
            turn, state = await self._agent.send_tool_results(results, state)
            totals = totals.merge(turn.usage)
        return self._presented(turn, totals, state, gate)

    def _emit_rewrites(self, screening: Screening) -> None:
        for rewrite in screening.rewrites:
            self._emit(
                ARGUMENT_NORMALIZED,
                call_id=rewrite.call_id,
                tool=rewrite.tool,
                argument=rewrite.argument,
                before=rewrite.before,
                after=rewrite.after,
            )

    # -- introspection (tests, logs) -------------------------------------------

    def rendered_prompts(self) -> dict[str, str]:
        """The exact system prompts this session's agent sends."""
        catalog = load_catalog(self.config.prompts_path, self.config.prompts.inline)
        prompts = {"backend": backend_system_prompt(catalog, self.config, self._prompt_context)}
        if self.config.frontend_enabled:
            key = self.config.frontend.prompt_key
            context = {**self._prompt_context, "in_progress": None}
            prompts["frontend"] = render(catalog.get(key), self.config, context, catalog=catalog, key=key)
        return prompts
