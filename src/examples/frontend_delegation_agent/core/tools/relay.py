# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend tool calls -> Realtime function calls or local callables, and results back (plan 8.2, 8.3).

* Calls arriving within ``batch_window_ms`` form one batch.
* ``ArgumentNormalizer.screen`` runs first (when configured): invalid or already-failed
  lookups are answered locally and never reach the client.
* ``ResultHints`` (when configured) appends a recovery note to a failed user or customer
  lookup's output before it goes to the gateway (I3, I4); with ``FieldProvenance`` the 2nd and
  later note also says which values the user spelled or only said (A3).
* ``FormatRetry`` (when configured) handles a not-found lookup by a phone field (A1 / A1-lite):
  on a tool that declares read-only semantics the
  relay looks up the same digits in the other format itself (a synthetic call ``<id>.fmt`` in its
  own batch, emitted after the original batch's acknowledgement; the original result is withheld
  until it settles); on any other read tool it appends a hint.
* :meth:`ToolRelay.reconfigure` swaps them once the session's tools are known (schema rules).
* ``executor: wire`` emits the batch as one Realtime response of ``function_call`` items;
  ``function_call_output`` goes back to the gateway, and a later ``response.create`` is the
  batch's acknowledgement (:meth:`ToolRelay.take_ack`).
* ``executor: local`` runs the calls in-process (browser demo), with an optional per-call delay.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_delegation_agent.core.engine.output_scheduler import CallsItem
from examples.frontend_delegation_agent.core.errors import WireProtocolError
from examples.frontend_delegation_agent.core.normalization.arguments import (
    ArgumentNormalizer,
    FailureKey,
    render_message,
)
from examples.frontend_delegation_agent.core.text.messages import ToolCall
from examples.frontend_delegation_agent.core.text.tools import ToolRegistry, ToolSpec
from examples.frontend_delegation_agent.core.tools.field_provenance import FieldProvenance
from examples.frontend_delegation_agent.core.tools.format_retry import Candidate, FormatRetry
from examples.frontend_delegation_agent.core.tools.identity_ledger import FOUND, IdentityLedger
from examples.frontend_delegation_agent.core.tools.result_hints import ResultHints

SendResult = Callable[[str, int, str, bool], None]  # call_id, epoch, output, local
Log = Callable[..., None]
#: Suffix of the relay's own format-retry call ids (A1).
RETRY_SUFFIX = ".fmt"

_BATCH_IDS = itertools.count(1)


@dataclass(slots=True)
class _Call:
    call_id: str
    epoch: int
    name: str
    arguments: str
    batch_id: str = ""
    key: FailureKey | None = None


@dataclass(slots=True)
class _Batch:
    batch_id: str
    call_ids: tuple[str, ...]
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(call_id in self.outputs for call_id in self.call_ids)


@dataclass(frozen=True, slots=True)
class FormatRetryMessages:
    """The catalog texts of ``format_retry``'s three message keys (``{value}``: the other form)."""

    hint: str = ""
    found: str = ""
    not_found: str = ""


@dataclass(slots=True)
class _Retry:
    """An A1 format retry in flight: the withheld original result and the synthetic call."""

    original: _Call
    output: str  # the original (not-found) output, withheld
    candidate: Candidate
    call: _Call
    emitted: bool = False
    timer: asyncio.TimerHandle | None = None


class ToolRelay:
    """One session's tool round trips."""

    def __init__(
        self,
        *,
        executor: str,
        send_result: SendResult,
        emit_calls: Callable[[CallsItem], None],
        log: Log,
        normalizer: ArgumentNormalizer | None = None,
        result_hints: ResultHints | None = None,
        local_tools: Sequence[ToolSpec] = (),
        batch_window_ms: int = 30,
        per_call_delay_s: float = 0.0,
        format_retry: FormatRetry | None = None,
        format_messages: FormatRetryMessages | None = None,
        provenance: FieldProvenance | None = None,
        result_timeout_s: float = 120.0,
    ) -> None:
        """Bind the session's sinks and settings."""
        self._executor = executor
        self._send_result = send_result
        self._emit_calls = emit_calls
        self._log = log
        self._normalizer = normalizer
        self._hints = result_hints
        self._registry = ToolRegistry(local_tools) if executor == "local" else None
        self._window = batch_window_ms / 1000.0
        self._delay = per_call_delay_s
        self._pending: list[_Call] = []
        self._flush_handle: asyncio.TimerHandle | None = None
        self._open: dict[str, _Call] = {}
        self._cancelled: set[str] = set()
        self._batches: list[_Batch] = []
        self._failed: frozenset[FailureKey] = frozenset()
        self._invalid_counts: dict[str, int] = {}  # tool -> invalid answers so far (escalating wording)
        self._tasks: set[asyncio.Task[None]] = set()
        self._format = format_retry
        self._format_messages = format_messages or FormatRetryMessages()
        self._provenance = provenance
        self._result_timeout_s = result_timeout_s
        self._retries: dict[str, _Retry] = {}  # synthetic call id -> retry

    # -- from the gateway -------------------------------------------------------------------

    def on_call(self, data: Mapping[str, Any]) -> None:
        """A ``tool.call`` from the gateway: start or join the current batch window."""
        call = _Call(
            call_id=str(data["call_id"]),
            epoch=int(data["epoch"]),
            name=str(data["name"]),
            arguments=str(data.get("arguments") or "{}"),
        )
        self._pending.append(call)
        if self._flush_handle is None:
            self._flush_handle = asyncio.get_running_loop().call_later(self._window, self._flush)

    def on_cancel(self, call_id: str, reason: str) -> None:
        """The gateway gave up on a call (timeout, interrupt, worker died)."""
        self._pending = [call for call in self._pending if call.call_id != call_id]
        call = self._open.pop(call_id, None)
        self._cancelled.add(call_id)
        if call is not None:
            self._mark_output(call_id, "")
        self._log("tool_cancelled", call_id=call_id, reason=reason)
        ledger = self._ledger
        if ledger is not None:
            ledger.on_cancel(call_id)
        for retry in [r for r in self._retries.values() if r.original.call_id == call_id]:
            self._drop_retry(retry, outcome="cancelled")

    # -- from the client --------------------------------------------------------------------

    def on_function_output(self, call_id: str, output: str) -> None:
        """A ``function_call_output`` item; unknown ids are a wire error, cancelled ones are dropped."""
        call = self._open.pop(call_id, None)
        if call is None:
            if call_id in self._cancelled:
                self._log("tool_result_ignored", call_id=call_id, reason="cancelled")
                return
            raise WireProtocolError(
                f"call_id {call_id!r} does not match an outstanding function call",
                code="invalid_value",
                param="item.call_id",
            )
        self._log("tool_output_in", call_id=call_id, output=output)
        retry = self._retries.get(call_id)
        if retry is not None:
            self._settle_retry(retry, output)
            return
        permanent = (
            call.key is not None and self._normalizer is not None and self._normalizer.is_permanent_failure(output)
        )
        if permanent and call.key is not None:
            self._failed = self._failed | {call.key}
        candidate = self._format_candidate(call) if permanent else None
        if candidate is not None and candidate.call:
            self._mark_output(call_id, output)
            self._start_retry(call, output, candidate)
            return
        if candidate is not None:
            output = self._format_hint(call, output, candidate)
        output = self._annotate(call, output)
        self._send_result(call_id, call.epoch, output, False)
        self._mark_output(call_id, output)

    def reconfigure(
        self,
        normalizer: ArgumentNormalizer | None,
        result_hints: ResultHints | None,
        format_retry: FormatRetry | None = None,
    ) -> None:
        """Use the collaborators rebuilt for new session tools (failed calls stay known)."""
        self._normalizer = normalizer
        self._hints = result_hints
        self._format = format_retry

    def take_ack(self) -> bool:
        """``response.create``: consume one tool batch's acknowledgement, if a batch awaits one."""
        for index, batch in enumerate(self._batches):
            if batch.complete or batch.outputs:
                del self._batches[index]
                self._log("tool_batch_ack", batch_id=batch.batch_id, complete=batch.complete)
                self._emit_ready_retries()
                return True
        if self._batches:
            batch = self._batches.pop(0)
            self._log("tool_batch_ack", batch_id=batch.batch_id, complete=False, early=True)
            self._emit_ready_retries()
            return True
        return False

    @property
    def outstanding(self) -> int:
        """Calls sent to the client and not answered yet."""
        return len(self._open)

    async def close(self) -> None:
        """Stop timers and local executions."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        for retry in list(self._retries.values()):
            if retry.timer is not None:
                retry.timer.cancel()
        self._retries.clear()
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # -- batching ---------------------------------------------------------------------------

    def _flush(self) -> None:
        self._flush_handle = None
        calls, self._pending = self._pending, []
        if not calls:
            return
        batch_id = f"batch_{next(_BATCH_IDS)}"
        sent = self._screen(calls)
        if not sent:
            return
        for call in sent:
            call.batch_id = batch_id
        ledger = self._ledger
        if ledger is not None:
            for call in sent:
                ledger.on_issue(call.call_id, call.name, call.arguments)
        if self._executor == "local":
            for call in sent:
                task = asyncio.create_task(self._run_local(call), name=f"local-tool-{call.call_id}")
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            return
        for call in sent:
            self._open[call.call_id] = call
        self._batches.append(_Batch(batch_id=batch_id, call_ids=tuple(call.call_id for call in sent)))
        self._log(
            "tool_calls_out",
            batch_id=batch_id,
            calls=[call.name for call in sent],
            call_ids=[call.call_id for call in sent],
            arguments=[call.arguments for call in sent],
        )
        self._emit_calls(CallsItem(batch_id=batch_id, calls=[(c.call_id, c.name, c.arguments) for c in sent]))

    def _screen(self, calls: list[_Call]) -> list[_Call]:
        """Canonicalize arguments and answer invalid / already-failed calls locally."""
        if self._normalizer is None:
            return calls
        screening = self._normalizer.screen(
            [ToolCall(id=call.call_id, name=call.name, arguments_json=call.arguments) for call in calls],
            self._failed,
            self._invalid_counts,
        )
        by_id = {call.call_id: call for call in calls}
        for rewrite in screening.rewrites:
            self._log(
                "argument_normalized",
                call_id=rewrite.call_id,
                tool=rewrite.tool,
                argument=rewrite.argument,
                before=rewrite.before,
                after=rewrite.after,
            )
        for answer in screening.local:
            call = by_id[answer.call_id]
            if answer.reason == "invalid":
                self._invalid_counts[answer.tool] = self._invalid_counts.get(answer.tool, 0) + 1
                if self._hints is not None:
                    self._hints.on_local_invalid(answer.tool, answer.argument)
                ledger = self._ledger
                if ledger is not None:
                    ledger.on_issue(call.call_id, call.name, call.arguments)
                    ledger.on_local_invalid(call.call_id)
            self._log(
                "call_answered_locally",
                call_id=answer.call_id,
                tool=answer.tool,
                reason=answer.reason,
                message_key=answer.message_key,
            )
            self._send_result(answer.call_id, call.epoch, answer.message, True)
        out: list[_Call] = []
        for canonical in screening.sent:
            call = by_id[canonical.id]
            call.arguments = canonical.arguments_json
            call.key = screening.keys.get(canonical.id)
            out.append(call)
        return out

    async def _run_local(self, call: _Call) -> None:
        assert self._registry is not None  # noqa: S101 - local executor only
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        result = await self._registry.execute(
            ToolCall(id=call.call_id, name=call.name, arguments_json=call.arguments or "{}")
        )
        self._log("tool_output_in", call_id=call.call_id, output=result.content, local_executor=True)
        self._send_result(call.call_id, call.epoch, self._annotate(call, result.content), False)

    @property
    def _ledger(self) -> IdentityLedger | None:
        return self._hints.ledger if self._hints is not None else None

    def _annotate(
        self,
        call: _Call,
        output: str,
        *,
        hint_output: str | None = None,
        ledger_outputs: Sequence[tuple[str, str]] | None = None,
    ) -> str:
        """The output with a recovery note appended when ``ResultHints`` asks for one.

        ``hint_output`` is what the hints judge (the lookup's own result), when ``output`` carries more.
        ``ledger_outputs`` (F8): the ``(call_id, output)`` pairs the ledger classifies, in order (an A1
        retry adds its own call); by default the call's own judged output.
        """
        if self._hints is None:
            return output
        judged = output if hint_output is None else hint_output
        reset, ledger_note = self._observe_identification(call, judged, ledger_outputs)
        hint = self._hints.on_output(call.name, judged, reset=reset)
        if ledger_note and hint is None:
            return f"{output.rstrip()}\n\n{ledger_note}"
        if hint is None:
            return output
        self._log("result_hint", call_id=call.call_id, tool=call.name, message_key=hint.message_key, n=hint.miss)
        note = hint.note
        if self._provenance is not None and self._hints.field_provenance and hint.miss >= 2:
            line = self._provenance.describe(call.arguments)
            if line:
                note = f"{note}\n{line}"
                self._log("field_provenance", call_id=call.call_id, tool=call.name, line=line)
        if ledger_note:
            note = f"{note}\n{ledger_note}"
        return f"{output.rstrip()}\n\n{note}"

    def _observe_identification(
        self, call: _Call, judged: str, ledger_outputs: Sequence[tuple[str, str]] | None
    ) -> tuple[bool | None, str]:
        """F8: classify identification outputs; ``(reset, note)`` for the hints and the appended text."""
        ledger = self._ledger
        if ledger is None:
            return None, ""
        found, note = False, ""
        for call_id, text in ledger_outputs if ledger_outputs is not None else ((call.call_id, judged),):
            output_class, ledger_note = ledger.on_output(call_id, text)
            if not output_class:
                continue
            self._log(
                "identity_ledger",
                call_id=call_id,
                tool=call.name,
                output_class=output_class,
                episode=ledger.episode,
                note_chars=len(ledger_note),
            )
            found = found or output_class == FOUND
            note = ledger_note or note
        if found:
            note = ""  # identified in this round: no ledger note
        return (found if ledger.watches(call.name) else False), note

    def _mark_output(self, call_id: str, output: str) -> None:
        for batch in self._batches:
            if call_id in batch.call_ids and call_id not in batch.outputs:
                batch.outputs[call_id] = output

    # -- format retry (A1 / A1-lite) ------------------------------------------------------------

    def _format_candidate(self, call: _Call) -> Candidate | None:
        if self._format is None or self._normalizer is None:
            return None
        return self._format.candidate(
            call.name, call.arguments, has_rule=self._normalizer.has_rule, failed=self._failed
        )

    def _format_hint(self, call: _Call, output: str, candidate: Candidate) -> str:
        """A1-lite: the not-found output with the other form, for the backend to look up once."""
        self._log_format(call, candidate, outcome="hint")
        note = render_message(self._format_messages.hint, value=candidate.after)
        return f"{output.rstrip()}\n\n{note}" if note else output

    def _start_retry(self, original: _Call, output: str, candidate: Candidate) -> None:
        """A1: withhold ``output`` and look up the other form (emitted after the original batch's ack)."""
        call = _Call(
            call_id=f"{original.call_id}{RETRY_SUFFIX}",
            epoch=original.epoch,
            name=original.name,
            arguments=candidate.arguments,
            key=candidate.key,
        )
        retry = _Retry(original=original, output=output, candidate=candidate, call=call)
        self._retries[call.call_id] = retry
        retry.timer = asyncio.get_running_loop().call_later(self._result_timeout_s, self._retry_timed_out, call.call_id)
        if not any(b.batch_id == original.batch_id for b in self._batches):
            self._emit_retry(retry)

    def _emit_ready_retries(self) -> None:
        """Emit retries whose original batch has been acknowledged."""
        pending = {batch.batch_id for batch in self._batches}
        for retry in list(self._retries.values()):
            if not retry.emitted and retry.original.batch_id not in pending:
                self._emit_retry(retry)

    def _emit_retry(self, retry: _Retry) -> None:
        retry.emitted = True
        call = retry.call
        ledger = self._ledger
        if ledger is not None:
            ledger.on_issue(call.call_id, call.name, call.arguments, origin="format_retry")
        call.batch_id = f"batch_{next(_BATCH_IDS)}"
        self._open[call.call_id] = call
        self._batches.append(_Batch(batch_id=call.batch_id, call_ids=(call.call_id,)))
        self._log(
            "tool_calls_out",
            batch_id=call.batch_id,
            calls=[call.name],
            call_ids=[call.call_id],
            arguments=[call.arguments],
            format_retry=True,
        )
        self._emit_calls(CallsItem(batch_id=call.batch_id, calls=[(call.call_id, call.name, call.arguments)]))

    def _settle_retry(self, retry: _Retry, output: str) -> None:
        """The other form's output: return the original call's result (outcome table, plan A1)."""
        del self._retries[retry.call.call_id]
        if retry.timer is not None:
            retry.timer.cancel()
        self._mark_output(retry.call.call_id, output)
        original, candidate = retry.original, retry.candidate
        assert self._normalizer is not None  # noqa: S101 - a retry starts only with a normalizer
        if self._normalizer.is_permanent_failure(output):
            self._failed = self._failed | {candidate.key}
            note = render_message(self._format_messages.not_found, value=candidate.after)
            combined = f"{retry.output.rstrip()}\n\n{note}" if note else retry.output
            pairs = ((original.call_id, retry.output), (retry.call.call_id, output))
            result = self._annotate(original, combined, hint_output=retry.output, ledger_outputs=pairs)
            outcome = "not_found"
        elif not output.strip() or output.lstrip().startswith("Error"):
            pairs = ((original.call_id, retry.output), (retry.call.call_id, output))
            result = self._annotate(
                original, retry.output, ledger_outputs=pairs
            )  # transient: the other form stays unmarked
            outcome = "transient"
        else:
            note = render_message(self._format_messages.found, value=candidate.after)
            combined = f"{note}\n{output}" if note else output
            pairs = ((original.call_id, retry.output), (retry.call.call_id, output))
            result = self._annotate(original, combined, hint_output=output, ledger_outputs=pairs)
            outcome = "found"
        self._log_format(original, candidate, outcome=outcome)
        self._send_result(original.call_id, original.epoch, result, False)

    def _retry_timed_out(self, call_id: str) -> None:
        retry = self._retries.get(call_id)
        if retry is None:
            return
        self._drop_retry(retry, outcome="timeout")
        self._send_result(
            retry.original.call_id, retry.original.epoch, self._annotate(retry.original, retry.output), False
        )

    def _drop_retry(self, retry: _Retry, *, outcome: str) -> None:
        """Forget a retry: its call is cancelled and a late output is ignored like any cancelled id."""
        self._retries.pop(retry.call.call_id, None)
        if retry.timer is not None:
            retry.timer.cancel()
        call_id = retry.call.call_id
        self._cancelled.add(call_id)
        if self._open.pop(call_id, None) is not None:
            self._mark_output(call_id, "")
        ledger = self._ledger
        if ledger is not None:
            ledger.on_cancel(call_id)
        self._log_format(retry.original, retry.candidate, outcome=outcome)

    def _log_format(self, call: _Call, candidate: Candidate, *, outcome: str) -> None:
        self._log(
            "format_retry",
            call_id=call.call_id,
            tool=candidate.tool,
            argument=candidate.argument,
            before=candidate.before,
            after=candidate.after,
            mode="call" if candidate.call else "hint",
            outcome=outcome,
        )
