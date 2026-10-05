# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Session history that the Frontend/Backend Thinker receives with each plan.

Two pieces:

* :class:`ConversationTranscript` records what was said. User text is the
  finalized turn text the Talker received. Assistant text comes from the
  Realtime conversation journal when one exists, because a client can truncate
  or delete an assistant item after it was spoken; otherwise it is the text
  Pipecat emitted or finalized for the turn.
* :class:`DelegationLedger` records each delegation: its request, the writes
  observed at their real side-effect boundary, the reads, how the run ended,
  and how its result was delivered. :meth:`DelegationLedger.render` returns a
  bounded JSON-ready view whose serialized size never exceeds
  :data:`MAX_HISTORY_CHARS`.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: A position in the conversation: (journal sequence, event counter).
Cursor = tuple[int, int]

MAX_ENTRIES = 8
MAX_TRANSCRIPT_MESSAGES = 6
MAX_TEXT_CHARS = 300
MAX_ARGUMENT_CHARS = 300
MAX_TOOL_NAME_CHARS = 64
MAX_HISTORY_CHARS = 6_000
_MAX_TRANSCRIPT_EVENTS = 64
_MAX_RETAINED_ENTRIES = 32

_WRITE_STATES = ("started", "confirmed", "unconfirmed")
_STATUSES = frozenset({"success", "partial", "not_found", "error", "unavailable"})
_NEEDS_INPUT_REASONS = frozenset(
    {
        "params_missing",
        "params_invalid",
        "params_optional",
        "confirmation_needed",
        "confirm_required",
        "past_date",
        "missing_search_context",
    }
)


class ConversationTranscript:
    """What was said in the session, in order, with roles kept.

    ``journal_provider`` returns the Realtime ``ConversationJournal`` (or None).
    With a journal, assistant text and typed user text are read from it when the
    history is rendered, so later truncations and deletions are followed. Spoken
    user text always comes from :meth:`record_user`, because the journal keeps a
    spoken transcript only when the client asked for input transcription.
    """

    def __init__(self, journal_provider: Callable[[], Any] | None = None) -> None:
        """Create an empty transcript, optionally backed by a Realtime journal."""
        self._journal_provider = journal_provider
        self._events: deque[tuple[Cursor, str, str]] = deque()
        self._counter = 0
        self._evicted: Cursor | None = None

    @property
    def follows_journal(self) -> bool:
        """Return whether assistant text comes from a Realtime journal."""
        return self._journal_provider is not None

    def record_user(self, text: object) -> None:
        """Record the finalized text of one user turn."""
        self._append("user", text)

    def record_assistant(self, text: object) -> None:
        """Record one finalized assistant turn; ignored when the journal is authoritative."""
        if not self.follows_journal:
            self._append("assistant", text)

    def cursor(self) -> Cursor:
        """Return the current position."""
        return (self._journal_sequence(), self._counter)

    def between(self, after: Cursor, upto: Cursor | None = None) -> tuple[list[dict[str, str]], bool]:
        """Return the messages said after ``after`` and up to ``upto``, and whether some are missing."""
        selected = [
            (key, role, text) for key, role, text in self._events if key > after and (upto is None or key <= upto)
        ]
        truncated = self._evicted is not None and self._evicted > after
        journal = self._journal()
        if journal is not None:
            oldest = journal.oldest_retained_sequence()
            if oldest is not None and oldest > after[0] + 1:
                truncated = True
            selected.extend(_journal_messages(journal, after, upto))
        selected.sort(key=lambda message: message[0])
        return [{"role": role, "text": text} for _key, role, text in selected], truncated

    def latest_user_text(self) -> str:
        """Return the newest user utterance, spoken or typed."""
        latest: tuple[Cursor, str] | None = None
        for key, role, text in reversed(self._events):
            if role == "user":
                latest = (key, text)
                break
        journal = self._journal()
        if journal is not None:
            floor: Cursor = (latest[0][0] - 1, 0) if latest is not None else (0, 0)
            for key, role, text in _journal_messages(journal, floor, None):
                if role == "user" and (latest is None or key > latest[0]):
                    latest = (key, text)
        return latest[1] if latest is not None else ""

    def reply_before_latest_user(self) -> tuple[str, str, int] | None:
        """Return the agent message the newest user turn answered, as the journal holds it now.

        Returns ``(text, status, user_turns)``: the newest assistant message
        before the newest user turn, its item status (``completed`` only when
        its response ended normally), and how many user turns followed it. A
        client that truncates the audio also cuts this text, so it is what the
        caller can have heard. None without a journal or without such a message.
        """
        journal = self._journal()
        if journal is None:
            return None
        live = set(journal.ordered_item_ids())
        assistants: list[tuple[Cursor, str, str]] = []
        users = [key for key, role, _text in self._events if role == "user"]
        for sequence, item_id in journal.added_after(0):
            if item_id not in live:
                continue
            item = journal.item(item_id)
            if item.get("type") != "message":
                continue
            key = (sequence, -1)
            if item.get("role") == "assistant":
                assistants.append((key, _journal_item_text(item, "assistant"), str(item.get("status") or "")))
            elif item.get("role") == "user" and _journal_item_text(item, "user"):
                users.append(key)
        if not users:
            return None
        latest_user = max(users)
        earlier = [entry for entry in assistants if entry[0] < latest_user]
        if not earlier:
            return None
        key, text, status = max(earlier, key=lambda entry: entry[0])
        return text, status, sum(1 for user in users if user > key)

    def _append(self, role: str, text: object) -> None:
        clean = " ".join(str(text or "").split())
        if not clean:
            return
        self._counter += 1
        self._events.append(((self._journal_sequence(), self._counter), role, clean))
        while len(self._events) > _MAX_TRANSCRIPT_EVENTS:
            evicted = self._events.popleft()[0]
            self._evicted = evicted if self._evicted is None else max(self._evicted, evicted)

    def _journal(self) -> Any:
        return self._journal_provider() if self._journal_provider is not None else None

    def _journal_sequence(self) -> int:
        journal = self._journal()
        return journal.last_sequence() if journal is not None else 0


def _journal_messages(journal: Any, after: Cursor, upto: Cursor | None) -> list[tuple[Cursor, str, str]]:
    """Resolve journal items added after ``after`` against the journal as it is now."""
    live = set(journal.ordered_item_ids())
    messages: list[tuple[Cursor, str, str]] = []
    for sequence, item_id in journal.added_after(after[0]):
        # An item added at sequence S sorts before user turns recorded at S.
        key = (sequence, -1)
        if upto is not None and key > upto:
            continue
        if item_id not in live:
            continue
        item = journal.item(item_id)
        if item.get("type") != "message":
            continue
        role = item.get("role")
        text = _journal_item_text(item, role)
        if text:
            messages.append((key, str(role), text))
    return messages


def _journal_item_text(item: Mapping[str, Any], role: object) -> str:
    """Return assistant output text, or typed user text; spoken user text is not taken from here."""
    wanted = {"assistant": ("output_audio", "output_text"), "user": ("input_text",)}.get(str(role))
    if wanted is None:
        return ""
    parts: list[str] = []
    for part in item.get("content") or ():
        if isinstance(part, Mapping) and part.get("type") in wanted:
            value = part.get("text") if str(part.get("type")).endswith("text") else part.get("transcript")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return " ".join(" ".join(parts).split())


@dataclass(slots=True)
class LedgerCall:
    """One tool call observed during a delegation."""

    tool: str
    kind: str
    arguments: str
    state: str | None = None
    status: str | None = None


@dataclass(slots=True)
class LedgerConfirmation:
    """One consent question the backend asked about a specific call.

    ``params`` are the canonical arguments the call would be sent with, and
    ``text`` is exactly what was handed to the Talker. ``summarized`` marks a
    question that could not state every value in full. ``outcome`` follows the
    question: ``pending`` until its response ends, then ``heard`` (the response
    completed without a reported truncation) or ``not_heard``; ``consumed``
    once its approval was used and ``expired`` once a later turn replaced it.
    """

    run_id: str
    tool: str
    params: dict[str, Any]
    text: str
    summarized: bool = False
    outcome: str = "pending"
    #: Session-update generation when it was asked; a later update voids the approval.
    generation: int = 0


@dataclass(slots=True)
class _Entry:
    number: int
    run_id: str
    request: str
    after: Cursor
    upto: Cursor
    interruptions_at_open: int
    calls: list[LedgerCall] = field(default_factory=list)
    result: str = "in_progress"
    result_text: str = ""
    delivery: str = ""
    delivered_text: str = ""
    interruption_mark: int | None = None
    confirmation: LedgerConfirmation | None = None
    #: The closing payload's ``reason`` and ``context``/``tool``; not rendered.
    outcome_reason: str = ""
    outcome_context: str = ""


class DelegationLedger:
    """Bounded record of the session's delegations for the Thinker."""

    def __init__(
        self,
        transcript: ConversationTranscript | None = None,
        *,
        interruption_count: Callable[[], int] | None = None,
    ) -> None:
        """Bind the ledger to the session transcript and barge-in counter."""
        self._transcript = transcript
        self._interruption_count = interruption_count
        self._entries: deque[_Entry] = deque(maxlen=_MAX_RETAINED_ENTRIES)
        self._by_run: dict[str, _Entry] = {}
        self._number = 0
        self._cursor: Cursor = (0, 0)

    def open(self, run_id: str, request: str) -> None:
        """Start the entry for a new backend run."""
        upto = self._transcript.cursor() if self._transcript is not None else (0, 0)
        self._number += 1
        entry = _Entry(
            number=self._number,
            run_id=run_id,
            request=request,
            after=self._cursor,
            upto=upto,
            interruptions_at_open=self._interruptions(),
        )
        self._cursor = upto
        if len(self._entries) == self._entries.maxlen:
            self._by_run.pop(self._entries[0].run_id, None)
        self._entries.append(entry)
        self._by_run[run_id] = entry

    def write_started(self, run_id: str | None, tool: str, arguments: object) -> LedgerCall | None:
        """Record a write as its side-effect boundary is entered."""
        entry = self._by_run.get(run_id or "")
        if entry is None:
            return None
        call = LedgerCall(
            tool=_clip(tool, MAX_TOOL_NAME_CHARS), kind="write", arguments=_arguments(arguments), state="started"
        )
        entry.calls.append(call)
        return call

    @staticmethod
    def write_confirmed(call: LedgerCall | None, status: object) -> None:
        """Record that a started write returned with ``status``."""
        if call is not None:
            call.state = "confirmed"
            call.status = _status(status)

    @staticmethod
    def write_unconfirmed(call: LedgerCall | None) -> None:
        """Record that a started write ended without a result; it may still have taken effect."""
        if call is not None and call.state == "started":
            call.state = "unconfirmed"

    def read(
        self, run_id: str | None, tool: object, arguments: object, status: object, *, state: str | None = None
    ) -> None:
        """Record one read and its status; ``state`` marks a read that never reached the tool (``answered_locally``)."""
        entry = self._by_run.get(run_id or "")
        if entry is not None:
            entry.calls.append(
                LedgerCall(
                    tool=_clip(str(tool or "tool"), MAX_TOOL_NAME_CHARS),
                    kind="read",
                    arguments=_arguments(arguments),
                    state=state,
                    status=_status(status),
                )
            )

    def confirmation(
        self,
        run_id: str | None,
        tool: str,
        params: Mapping[str, Any],
        text: str,
        *,
        summarized: bool = False,
    ) -> LedgerConfirmation | None:
        """Record the consent question a run asked; the newest question replaces older ones."""
        entry = self._by_run.get(run_id or "")
        if entry is None:
            return None
        record = LedgerConfirmation(
            run_id=entry.run_id,
            tool=_clip(tool, MAX_TOOL_NAME_CHARS),
            params=dict(params),
            text=text,
            summarized=summarized,
        )
        entry.confirmation = record
        return record

    @property
    def transcript(self) -> ConversationTranscript | None:
        """Return the session transcript this ledger reads, if any."""
        return self._transcript

    def latest_confirmation(self) -> LedgerConfirmation | None:
        """Return the consent question of the newest entry that asked one."""
        for entry in reversed(self._entries):
            if entry.confirmation is not None:
                return entry.confirmation
        return None

    def entry_writes(self, run_id: str | None) -> list[LedgerCall]:
        """Return the writes recorded for one run, oldest first."""
        entry = self._by_run.get(run_id or "")
        return [call for call in entry.calls if call.kind == "write"] if entry is not None else []

    def entry_calls(self, run_id: str | None) -> list[LedgerCall]:
        """Return every call recorded for one run, oldest first."""
        entry = self._by_run.get(run_id or "")
        return list(entry.calls) if entry is not None else []

    def outcome(self, run_id: str | None) -> tuple[str, str, str] | None:
        """Return ``(result, reason, context)`` of a closed run, or None while it is open or unknown."""
        entry = self._by_run.get(run_id or "")
        if entry is None or entry.result == "in_progress":
            return None
        return entry.result, entry.outcome_reason, entry.outcome_context

    def runs_after(self, run_id: str | None) -> list[str]:
        """Return the run ids opened after ``run_id``, oldest first."""
        entries = list(self._entries)
        for index, entry in enumerate(entries):
            if entry.run_id == run_id:
                return [later.run_id for later in entries[index + 1 :]]
        return []

    def close(self, run_id: str, payload: Mapping[str, Any]) -> None:
        """Record how a run ended, classified from its payload."""
        entry = self._by_run.get(run_id)
        if entry is None:
            return
        entry.result = classify_result(payload)
        entry.result_text = _clip(str(payload.get("response_text") or ""), MAX_TEXT_CHARS)
        entry.outcome_reason = str(payload.get("reason") or "")
        entry.outcome_context = str(payload.get("context") or payload.get("tool") or "")
        self._settle_started(entry)

    def cancelled(self, run_id: str) -> None:
        """Record a run cancelled before it produced a payload."""
        entry = self._by_run.get(run_id)
        if entry is None:
            return
        entry.result = "cancelled"
        self._settle_started(entry)

    def record_delivery(self, run_id: str | None, delivery: str, delivered_text: str = "") -> None:
        """Record how the run's result reached (or did not reach) the user; the last caller wins."""
        entry = self._by_run.get(run_id or "")
        if entry is None:
            return
        entry.delivery = delivery
        if delivered_text and self._transcript is not None and not self._transcript.follows_journal:
            entry.delivered_text = _clip(delivered_text, MAX_TEXT_CHARS)
            entry.interruption_mark = self._interruptions()
        else:
            entry.delivered_text = ""
            entry.interruption_mark = None

    def render(self, current_run_id: str | None = None) -> list[dict[str, Any]] | dict[str, Any] | None:
        """Return the bounded history, or None while no earlier delegation exists."""
        entries = list(self._entries)
        if not any(entry.run_id != current_run_id for entry in entries):
            return None
        older, entries = entries[:-MAX_ENTRIES], entries[-MAX_ENTRIES:]
        omitted = len(older) + (self._number - len(self._entries))
        prior = [{"tool_calls": [_render_call(call) for call in entry.calls]} for entry in older]
        rendered = [self._render_entry(entry, index, entries, current_run_id) for index, entry in enumerate(entries)]
        return _fit(rendered, omitted, prior)

    def _render_entry(
        self, entry: _Entry, index: int, entries: list[_Entry], current_run_id: str | None
    ) -> dict[str, Any]:
        transcript: list[dict[str, str]] = []
        truncated = False
        if self._transcript is not None:
            transcript, truncated = self._transcript.between(entry.after, entry.upto)
        if len(transcript) > MAX_TRANSCRIPT_MESSAGES:
            transcript = transcript[-MAX_TRANSCRIPT_MESSAGES:]
            truncated = True
        rendered: dict[str, Any] = {"run": entry.number}
        if entry.run_id == current_run_id:
            rendered["current"] = True
        rendered["transcript"] = [
            {"role": message["role"], "text": _clip(message["text"], MAX_TEXT_CHARS)} for message in transcript
        ]
        if truncated:
            rendered["history_truncated"] = True
        rendered["request"] = _clip(entry.request, MAX_TEXT_CHARS)
        rendered["tool_calls"] = [_render_call(call) for call in entry.calls]
        rendered["result"] = entry.result
        if entry.result_text:
            rendered["result_text"] = entry.result_text
        if entry.delivery:
            rendered["delivery"] = _delivery(entry)
        if entry.confirmation is not None:
            rendered["confirmation"] = {
                "tool": entry.confirmation.tool,
                "arguments": _arguments(entry.confirmation.params),
                "state": entry.confirmation.outcome,
            }
        if entry.delivered_text:
            rendered["delivered_text"] = entry.delivered_text
            following = entries[index + 1].interruptions_at_open if index + 1 < len(entries) else self._interruptions()
            if entry.interruption_mark is not None and following > entry.interruption_mark:
                rendered["possibly_interrupted"] = True
        return rendered

    def _interruptions(self) -> int:
        return self._interruption_count() if self._interruption_count is not None else 0

    @staticmethod
    def _settle_started(entry: _Entry) -> None:
        for call in entry.calls:
            if call.state == "started":
                call.state = "unconfirmed"


def classify_result(payload: Mapping[str, Any]) -> str:
    """Return how a run ended, from its payload rather than from exceptions."""
    if payload.get("type") == "tool_result":
        status = str(payload.get("status") or "error")
        return "answered" if status in {"success", "partial", "not_found"} else "failed"
    reason = str(payload.get("reason") or "")
    if reason in _NEEDS_INPUT_REASONS:
        return "needs_input"
    if reason == "timeout":
        return "timeout"
    if reason in {"planner_error", "planner_error_exhausted"}:
        return "planner_failed"
    if reason in {"unsupported_request", "tool_disabled"}:
        return "unsupported"
    if reason in {"aborted", "cancelled", "booking_cancelled"}:
        return "cancelled"
    if reason == "tool_error":
        return "failed"
    return "answered"


def _render_call(call: LedgerCall) -> dict[str, Any]:
    rendered: dict[str, Any] = {"tool": call.tool, "kind": call.kind, "arguments": call.arguments}
    if call.state is not None:
        rendered["state"] = call.state
    if call.status is not None:
        rendered["status"] = call.status
    return rendered


def _delivery(entry: _Entry) -> str:
    if entry.delivery == "not_delivered":
        return "not_delivered_cancelled" if entry.result == "cancelled" else "not_delivered_superseded"
    return entry.delivery


def _fit(
    entries: list[dict[str, Any]], omitted: int, prior: list[dict[str, Any]]
) -> list[dict[str, Any]] | dict[str, Any]:
    """Apply the compaction steps in order until the history fits ``MAX_HISTORY_CHARS``.

    ``prior`` holds the tool calls of older entries that are not rendered; their
    writes are still counted once the output takes the truncated form.
    """
    dropped = list(prior)

    def view() -> list[dict[str, Any]] | dict[str, Any]:
        return _summary(omitted, dropped, entries) if omitted else entries

    def fits() -> bool:
        return _size(view()) <= MAX_HISTORY_CHARS

    if fits():
        return view()
    # 1. Transcript text; 2-5. read arguments, result texts, write arguments,
    # then read lists. Each step runs oldest entry first.
    steps: tuple[Callable[[dict[str, Any]], bool], ...] = (
        _drop_transcript,
        lambda entry: _drop_arguments(entry, "read"),
        _drop_texts,
        lambda entry: _drop_arguments(entry, "write"),
        _count_reads,
    )
    for step in steps:
        for entry in entries:
            if step(entry):
                entry["history_truncated"] = True
                if fits():
                    return view()
    # 6. Whole oldest entries; their writes stay in the fixed-key counts.
    while entries:
        dropped.append(entries.pop(0))
        omitted += 1
        if entries and fits():
            return view()
    # 7. Final fallback: fixed keys, write counts without tool names, newest writes that fit.
    return _fallback(omitted, dropped)


def _summary(omitted: int, dropped: list[dict[str, Any]], entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "history_truncated": True,
        "omitted_entries": omitted,
        "write_counts": _write_counts(dropped),
        "entries": entries,
    }


def _fallback(omitted: int, dropped: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "history_truncated": True,
        "omitted_entries": omitted,
        "write_counts": _write_counts(dropped),
        "recent_writes": [],
    }
    for write in reversed(list(_writes(dropped))):
        candidate = {key: write[key] for key in ("tool", "state", "status") if key in write}
        result["recent_writes"].append(candidate)
        if _size(result) > MAX_HISTORY_CHARS:
            result["recent_writes"].pop()
            break
    return result


def _writes(entries: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for entry in entries:
        for call in entry.get("tool_calls") or ():
            if call.get("kind") == "write":
                yield call


def _write_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for call in _writes(entries):
        key = f"{call.get('state') or 'unknown'}/{call.get('status') or 'none'}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _drop_transcript(entry: dict[str, Any]) -> bool:
    if not entry.get("transcript"):
        return False
    entry["transcript"] = []
    return True


def _drop_arguments(entry: dict[str, Any], kind: str) -> bool:
    changed = False
    confirmation = entry.get("confirmation")
    if kind == "write" and isinstance(confirmation, dict) and "arguments" in confirmation:
        del confirmation["arguments"]
        changed = True
    for call in entry.get("tool_calls") or ():
        if call.get("kind") == kind and "arguments" in call:
            del call["arguments"]
            changed = True
    return changed


def _drop_texts(entry: dict[str, Any]) -> bool:
    changed = False
    for key in ("result_text", "delivered_text"):
        if key in entry:
            del entry[key]
            changed = True
    return changed


def _count_reads(entry: dict[str, Any]) -> bool:
    calls = entry.get("tool_calls") or []
    reads = [call for call in calls if call.get("kind") == "read"]
    if not reads:
        return False
    entry["tool_calls"] = [call for call in calls if call.get("kind") != "read"]
    entry["read_count"] = entry.get("read_count", 0) + len(reads)
    return True


def _status(status: object) -> str:
    value = str(status or "").strip().lower()
    return value if value in _STATUSES else "other"


def _arguments(arguments: object) -> str:
    try:
        text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(arguments)
    return _clip(text, MAX_ARGUMENT_CHARS)


def _clip(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def _size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False))
