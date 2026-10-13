# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-turn latency breakdown and tool-call rows, derived from the recorded timeline.

``derive`` is a pure function from a session's recorded rows (turns, events,
metrics, LLM calls, system samples) to ``turn_metrics`` and ``tool_calls`` rows.
The live recorder runs it in its writer thread when a turn ends and when the
session closes; the backfill runs it over stored sessions. Both go through
``recompute_session``, so old and new sessions are computed the same way.

Latencies
---------
- **Voice latency**: the user stopped speaking -> first bot audio.
- **Response latency**: the user stopped speaking -> first *perceivable* response,
  that is the earlier of the first bot audio and the send time of the first
  perceivable tool call of the turn. Equal to the voice latency when the turn has
  no perceivable action.

Stages
------
The interval from the end of user speech to the first response is cut along the
critical path. Each stage owns a stretch of that timeline, stretches never
overlap, and ``unexplained_secs`` is whatever no stage owns, so the stages plus
the remainder always add up to ``total_secs``:

- ``asr``: end of speech -> final transcript.
- ``turn_detection``: the rest of the wait until the turn is released to the LLM
  (VAD silence window, smart-turn decision). ASR finalization and turn detection
  run concurrently; the time is attributed to whichever finished last.
- ``intent_match``: intent engine matching (only when the engine is on).
- ``llm_first``: first LLM call: start -> first token, or start -> tool call sent
  when the call ends in a tool call.
- ``tool``: tool round trips (call sent -> result) and intent target calls.
- ``llm_later``: every later LLM call up to the first token of the speaking one.
- ``text_aggregation``: first token -> first sentence handed to the TTS.
- ``tts``: TTS time to first byte.

Turn kind
---------
``intent`` (the intent engine answered) > ``vision`` (an LLM call of the turn had
image input) > ``tool`` (at least one tool or intent call) > ``plain``.

Repairing older recordings
--------------------------
Recorders before ``RECORDER_VERSION`` 2 stored the text of an interrupted
assistant response (with ``interrupted`` and ``bot_stopped_at``) on the *next*
turn. ``--repair-attribution`` moves it back to the turn that opened the response
and then recomputes the metrics. Each session is repaired once: an audit
annotation (kind ``turn_attribution_repair``) lists what moved and marks the
session as done.

Usage::

    uv run python -m monitoring.turn_metrics                 # backfill every session
    uv run python -m monitoring.turn_metrics --session <id>  # one session
    uv run python -m monitoring.turn_metrics --repair-attribution --dry-run
    uv run python -m monitoring.turn_metrics --repair-attribution
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Any

from sqlalchemy import select

from monitoring import schema
from monitoring.store import SessionStore

STAGES = (
    "turn_detection",
    "asr",
    "intent_match",
    "llm_first",
    "tool",
    "llm_later",
    "text_aggregation",
    "tts",
)
KINDS = ("plain", "tool", "intent", "vision")
OUTCOMES = ("ok", "error", "timeout", "cancelled")

# Stored in the session snapshot. 2: an interrupted response stays on its own turn.
RECORDER_VERSION = 2
REPAIR_ANNOTATION = "turn_attribution_repair"
# An interrupted response is recognized on the next turn when that turn's text starts
# with at least this much of the previous turn's last LLM output.
_REPAIR_MIN_CHARS = 12

# Session snapshot key listing the client tools declared with ``perceivable: true``.
PERCEIVABLE_TOOLS_KEY = "client_tools_perceivable"
INTENT_EVENT = "intent_engine"
INTENT_MATCH_METRIC = "intent_match_secs"
INTENT_TARGET_METRIC = "intent_target_secs"
INTENT_CALL_ID_PREFIX = "intent-"
HOME_ASSISTANT_TARGET = "home_assistant"
# Results the client-tool timeout handler and pipecat's own timeout produce.
TIMEOUT_ERROR_PREFIX = "Timed out"

# Slack when matching events that are recorded a few milliseconds apart.
_EPS = 0.05
_METRIC_NAMES = ("ttfb", "text_aggregation", INTENT_MATCH_METRIC, INTENT_TARGET_METRIC)
_LLM_CALL_COLUMNS = (
    "id",
    "turn_idx",
    "started_at",
    "ended_at",
    "function_calls",
    "ttfb",
    "prompt_tokens",
    "completion_tokens",
    "n_images",
    "interrupted",
)


def service_role(processor: str | None) -> str:
    """ASR / LLM / TTS for pipecat service names (instance suffix ignored), else the bare name."""
    service = re.sub(r"#\d+$", "", processor or "?")
    for marker, role in (("STT", "ASR"), ("ASR", "ASR"), ("LLM", "LLM"), ("TTS", "TTS")):
        if marker in service:
            return role
    return service


def call_outcome(result: Any) -> str:
    """Classify a tool result as ``ok``, ``error`` or ``timeout``."""
    if result is None:
        return "timeout"  # pipecat's function-call timeout reports a None result
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str) and error.startswith(TIMEOUT_ERROR_PREFIX):
            return "timeout"
        if error or result.get("status") == "error":
            return "error"
    return "ok"


# ---------------------------------------------------------------------- loading
def load_trace(store: SessionStore, session_id: str, *, since: float | None = None) -> dict[str, Any] | None:
    """Read what ``derive`` needs for a session (rows older than ``since`` are skipped)."""
    session = store.get_session(session_id)
    if session is None:
        return None
    e, m, c = schema.events, schema.metrics, schema.llm_calls
    events_stmt = select(e).where(e.c.session_id == session_id).order_by(e.c.ts, e.c.id)
    metrics_stmt = (
        select(m.c.ts, m.c.processor, m.c.name, m.c.value)
        .where(m.c.session_id == session_id, m.c.name.in_(_METRIC_NAMES))
        .order_by(m.c.ts, m.c.id)
    )
    calls_stmt = (
        select(*(c.c[name] for name in _LLM_CALL_COLUMNS)).where(c.c.session_id == session_id).order_by(c.c.started_at)
    )
    if since is not None:
        events_stmt = events_stmt.where(e.c.ts >= since)
        metrics_stmt = metrics_stmt.where(m.c.ts >= since)
        calls_stmt = calls_stmt.where(c.c.started_at >= since)
    with store.engine.connect() as conn:
        events = [dict(r) for r in conn.execute(events_stmt).mappings()]
        metrics = [dict(r) for r in conn.execute(metrics_stmt).mappings()]
        llm_calls = [dict(r) for r in conn.execute(calls_stmt).mappings()]
    turns = store.rows("turns", session_id)
    start = since if since is not None else session["started_at"]
    end = session.get("ended_at") or time.time()
    return {
        "session": session,
        "turns": turns,
        "events": events,
        "metrics": metrics,
        "llm_calls": llm_calls,
        "samples": store.system_samples_between(start - 1.0, end + 1.0),
    }


# -------------------------------------------------------------------- deriving
def _tool_calls(trace: dict[str, Any], perceivable: set[str]) -> list[dict[str, Any]]:
    """Build one row per tool or intent call (without ``turn_idx``, assigned by the caller)."""
    session_id = trace["session"]["id"]
    config = trace["session"].get("config") or {}
    llm_ids = {
        call.get("tool_call_id") for row in trace["llm_calls"] for call in row.get("function_calls") or [] if call
    }
    calls: dict[str, dict[str, Any]] = {}

    def started(call_id: str | None, name: str | None, ts: float) -> None:
        if not call_id or call_id in calls:
            return
        from_intent = call_id not in llm_ids and call_id.startswith(INTENT_CALL_ID_PREFIX)
        calls[call_id] = {
            "session_id": session_id,
            "call_id": call_id,
            "name": name or "?",
            "trigger": "intent" if from_intent else "llm",
            "target": "client",
            "sent_at": ts,
            "duration_secs": None,
            "outcome": "cancelled",  # until a result shows up
            "perceivable": (name or "") in perceivable,
            "_open": True,
        }

    intent_metrics = [m for m in trace["metrics"] if m["name"] == INTENT_TARGET_METRIC]
    intent_timeout = (config.get("intent_engine") or {}).get("timeout_secs")
    for event in trace["events"]:
        kind, data, ts = event["kind"], event.get("data") or {}, event["ts"]
        if kind == "function_call_in_progress":
            started(data.get("tool_call_id"), data.get("name"), ts)
        elif kind == "function_calls":
            # Recordings made before ``function_call_in_progress`` existed: the batch
            # event is the closest thing to a send time.
            for call in data.get("calls") or []:
                started(call.get("tool_call_id"), call.get("name"), ts)
        elif kind == "function_result":
            call = calls.get(data.get("tool_call_id"))
            if call is not None and call["_open"]:  # the first result wins (a late client answer is ignored)
                call["_open"] = False
                call["duration_secs"] = max(0.0, ts - call["sent_at"])
                call["outcome"] = call_outcome(data.get("result"))
        elif kind == "function_call_cancelled":
            call = calls.get(data.get("tool_call_id"))
            if call is not None and call["_open"]:
                call["_open"] = False
                call["duration_secs"] = max(0.0, ts - call["sent_at"])
        elif kind == INTENT_EVENT and data.get("target") == HOME_ASSISTANT_TARGET and data.get("reason") != "dry_run":
            # Home Assistant calls go over HTTP: no function-call frames. The engine
            # reports the target time just before the event.
            timing = next((m for m in reversed(intent_metrics) if ts - 1.0 <= m["ts"] <= ts + _EPS), None)
            duration = timing["value"] if timing else None
            if data.get("handled_by") == "intent":
                outcome = "error" if data.get("response_type") == "error" else "ok"
            elif duration is not None and intent_timeout and duration >= float(intent_timeout) - _EPS:
                outcome = "timeout"
            else:
                outcome = "error"
            call_id = f"{INTENT_CALL_ID_PREFIX}ha-{event.get('id', int(ts * 1000))}"
            calls[call_id] = {
                "session_id": session_id,
                "call_id": call_id,
                "name": data.get("intent") or "?",
                "trigger": "intent",
                "target": HOME_ASSISTANT_TARGET,
                "sent_at": (timing["ts"] - duration) if timing else ts,
                "duration_secs": duration,
                "outcome": outcome,
                "perceivable": False,  # Home Assistant actions are never perceivable (v1)
                "_open": False,
            }
    return sorted(calls.values(), key=lambda call: call["sent_at"])


def _barge_ins(events: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    """``(ts, bot_was_speaking)`` for every user speech start, in order."""
    speaking, out = False, []
    for event in events:
        if event["kind"] == "bot_started_speaking":
            speaking = True
        elif event["kind"] == "bot_stopped_speaking":
            speaking = False
        elif event["kind"] == "user_started_speaking":
            out.append((event["ts"], speaking))
    return out


def _segments(raw: list[tuple[str, float, float]], start: float, end: float) -> list[dict[str, Any]]:
    """Clip stage stretches to ``[start, end]`` and make them non-overlapping (earlier start wins)."""
    cursor, out = start, []
    for stage, seg_start, seg_end in sorted(raw, key=lambda s: (s[1], s[2])):
        seg_start, seg_end = max(seg_start, cursor), min(seg_end, end)
        if seg_end > seg_start:
            out.append({"stage": stage, "start": seg_start, "end": seg_end})
            cursor = seg_end
    return out


def _turn_row(
    turn: dict[str, Any],
    window: tuple[float, float],
    trace: dict[str, Any],
    calls: list[dict[str, Any]],
    barge_in: bool | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(turn_metrics row, turns column updates)`` for one user turn."""
    w_start, w_end = window
    events = [e for e in trace["events"] if w_start <= e["ts"] < w_end]
    metrics = [m for m in trace["metrics"] if w_start <= m["ts"] < w_end]
    llm_calls = [c for c in trace["llm_calls"] if w_start <= c["started_at"] < w_end]

    def data(event: dict[str, Any]) -> dict[str, Any]:
        return event.get("data") or {}

    # ---- anchors
    released = turn.get("user_stopped_at")
    speech_end = turn.get("user_speech_stopped_at")
    if speech_end is None:
        # Older recordings: pipecat's latency breakdown carries the VAD-adjusted time.
        speech_end = next(
            (
                data(e)["user_turn_start_time"]
                for e in events
                if e["kind"] == "latency_breakdown" and data(e).get("user_turn_start_time")
            ),
            None,
        )
    audio = turn.get("bot_speech_started_at")
    if audio is None:
        floor = released if released is not None else w_start
        audio = next((e["ts"] for e in events if e["kind"] == "bot_started_speaking" and e["ts"] >= floor - _EPS), None)
    if speech_end is not None and audio is not None and audio < speech_end:
        audio = None

    perceivable_at = None
    if speech_end is not None:
        sends = [c["sent_at"] for c in calls if c["perceivable"] and c["sent_at"] >= speech_end]
        perceivable_at = min(sends, default=None)
    voice = audio - speech_end if speech_end is not None and audio is not None else None
    response_at = min((t for t in (audio, perceivable_at) if t is not None), default=None)
    response = response_at - speech_end if speech_end is not None and response_at is not None else None
    response_via = None
    if response_at is not None:
        response_via = "audio" if response_at == audio else "tool"

    # ---- stage stretches up to the first response
    stages: dict[str, float | None] = dict.fromkeys(STAGES)
    segments: list[dict[str, Any]] = []
    total = unexplained = None
    end = audio if audio is not None else response_at
    if speech_end is not None and end is not None:
        raw: list[tuple[str, float, float]] = []
        release = min(released, end) if released is not None else None
        if release is not None and release > speech_end:
            finals = [e["ts"] for e in events if e["kind"] == "asr_final" and speech_end <= e["ts"] <= release + _EPS]
            asr_end = min(max(finals), release) if finals else speech_end
            raw.append(("asr", speech_end, asr_end))
            raw.append(("turn_detection", asr_end, release))
        for metric in metrics:
            if metric["name"] == INTENT_MATCH_METRIC:
                raw.append(("intent_match", metric["ts"] - metric["value"], metric["ts"]))
            elif metric["name"] == INTENT_TARGET_METRIC:
                raw.append(("tool", metric["ts"] - metric["value"], metric["ts"]))
        path = [c for c in llm_calls if c["started_at"] < end]
        first_token = None
        for i, call in enumerate(path):
            stage = "llm_first" if call is llm_calls[0] else "llm_later"
            if i + 1 == len(path):
                # The call the first response comes from: its own text is spoken (first
                # token), or its tool call is the perceivable response (whole call).
                if audio is None:
                    raw.append((stage, call["started_at"], end))
                elif call.get("ttfb"):
                    first_token = call["started_at"] + call["ttfb"]
                    raw.append((stage, call["started_at"], first_token))
                continue
            next_start = path[i + 1]["started_at"]
            ids = {f.get("tool_call_id") for f in call.get("function_calls") or []}
            tools = [c for c in calls if c["call_id"] in ids] or [
                c for c in calls if c["trigger"] == "llm" and call["started_at"] <= c["sent_at"] < next_start
            ]
            if tools:
                sent = min(c["sent_at"] for c in tools)
                done = max(
                    (c["sent_at"] + c["duration_secs"] if c["duration_secs"] is not None else next_start) for c in tools
                )
                raw.append((stage, call["started_at"], sent))
                raw.append(("tool", sent, min(done, next_start)))
            else:
                # Superseded before speaking (interrupted, or re-triggered by a late result).
                raw.append((stage, call["started_at"], min(call.get("ended_at") or next_start, next_start)))
        tail = first_token if first_token is not None else (release if release is not None else speech_end)
        late = [m for m in metrics if tail < m["ts"] <= end + _EPS]
        aggregation = next((m for m in late if m["name"] == "text_aggregation"), None)
        if aggregation:
            raw.append(("text_aggregation", max(aggregation["ts"] - aggregation["value"], tail), aggregation["ts"]))
        tts = next(
            (m for m in late if m["name"] == "ttfb" and m["value"] > 0 and service_role(m["processor"]) == "TTS"),
            None,
        )
        if tts:
            raw.append(("tts", tts["ts"] - tts["value"], tts["ts"]))
        segments = _segments(raw, speech_end, end)
        total = end - speech_end
        for stage in STAGES:
            lengths = [s["end"] - s["start"] for s in segments if s["stage"] == stage]
            stages[stage] = sum(lengths) if lengths else None
        unexplained = max(0.0, total - sum(s["end"] - s["start"] for s in segments))

    # ---- kind, counts, load
    intent_handled = any(e["kind"] == INTENT_EVENT and data(e).get("handled_by") == "intent" for e in events)
    n_images = sum(c.get("n_images") or 0 for c in llm_calls)
    if intent_handled:
        kind = "intent"
    elif n_images:
        kind = "vision"
    elif calls:
        kind = "tool"
    else:
        kind = "plain"
    prompt = [c["prompt_tokens"] for c in llm_calls if c.get("prompt_tokens") is not None]
    completion = [c["completion_tokens"] for c in llm_calls if c.get("completion_tokens") is not None]
    bot_stops = [e["ts"] for e in events if e["kind"] == "bot_stopped_speaking" and e["ts"] > (audio or w_start)]
    load_end = max(bot_stops) if bot_stops else w_end
    loads = [s["gpu_load"] for s in trace["samples"] if w_start <= s["ts"] <= load_end and s["gpu_load"] is not None]

    row = {
        "session_id": turn["session_id"],
        "idx": turn["idx"],
        "kind": kind,
        "barge_in": barge_in,
        "voice_latency": voice,
        "response_latency": response,
        "response_via": response_via,
        **{f"{stage}_secs": value for stage, value in stages.items()},
        "unexplained_secs": unexplained,
        "total_secs": total,
        "segments": segments,
        "n_llm_calls": len(llm_calls),
        "n_tool_calls": len(calls),
        "n_images": n_images,
        "prompt_tokens": sum(prompt) if prompt else None,
        "completion_tokens": sum(completion) if completion else None,
        "gpu_load_mean": sum(loads) / len(loads) if loads else None,
        "gpu_load_peak": max(loads) if loads else None,
        "computed_at": time.time(),
    }
    updates: dict[str, Any] = {}
    if barge_in is not None:
        updates["barge_in"] = barge_in
    if turn.get("user_speech_stopped_at") is None and speech_end is not None:
        updates["user_speech_stopped_at"] = speech_end
    if turn.get("bot_speech_started_at") is None and audio is not None:
        updates["bot_speech_started_at"] = audio
    return row, updates


def derive(
    trace: dict[str, Any], *, only: set[int] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Return ``(turn_metrics rows, tool_calls rows, {turn idx: turns column updates})``.

    ``only`` restricts the output to those turn indexes (the live per-turn path).
    Turn 0 (the greeting) gets no ``turn_metrics`` row: it has no user speech.
    """
    config = trace["session"].get("config") or {}
    perceivable = {name for name in config.get(PERCEIVABLE_TOOLS_KEY) or [] if isinstance(name, str)}
    session = trace["session"]
    user_turns = sorted((t for t in trace["turns"] if t.get("user_started_at") is not None), key=lambda t: t["idx"])
    session_end = session.get("ended_at") or float("inf")
    windows: dict[int, tuple[float, float]] = {}
    for i, turn in enumerate(user_turns):
        end = user_turns[i + 1]["user_started_at"] if i + 1 < len(user_turns) else session_end
        # The speech events that open a turn are recorded a few milliseconds before the turn row.
        windows[turn["idx"]] = (turn["user_started_at"] - _EPS, end - _EPS)
    first_start = user_turns[0]["user_started_at"] if user_turns else session_end

    def turn_of(ts: float) -> int:
        for idx, (start, end) in windows.items():
            if start <= ts < end:
                return idx
        return 0 if ts < first_start else (user_turns[-1]["idx"] if user_turns else 0)

    calls = _tool_calls(trace, perceivable)
    for call in calls:
        call.pop("_open")
        call["turn_idx"] = turn_of(call["sent_at"])
    starts = _barge_ins(trace["events"])

    turn_rows, updates = [], {}
    for turn in user_turns:
        idx = turn["idx"]
        if only is not None and idx not in only:
            continue
        started = turn["user_started_at"]
        near = [(abs(ts - started), flag) for ts, flag in starts if abs(ts - started) < 0.5]
        barge_in = min(near)[1] if near else None
        row, turn_updates = _turn_row(turn, windows[idx], trace, [c for c in calls if c["turn_idx"] == idx], barge_in)
        turn_rows.append(row)
        updates[idx] = turn_updates
    if only is not None:
        calls = [c for c in calls if c["turn_idx"] in only]
    return turn_rows, calls, updates


# ------------------------------------------------------------------- persisting
def recompute_session(store: SessionStore, session_id: str, *, only_turn: int | None = None) -> int:
    """Derive and store a session's turn metrics; returns the number of turn rows written.

    ``only_turn`` recomputes a single finished turn (live path): rows before the
    previous turn are not read, and the other turns' rows are left untouched.
    """
    since = None
    if only_turn is not None:
        starts = {t["idx"]: t.get("user_started_at") for t in store.rows("turns", session_id)}
        since = starts.get(only_turn - 1) or starts.get(only_turn)
        if since is not None:
            since -= 1.0
    trace = load_trace(store, session_id, since=since)
    if trace is None:
        return 0
    turn_rows, call_rows, updates = derive(trace, only={only_turn} if only_turn is not None else None)
    store.write_derived(
        session_id, turn_rows=turn_rows, call_rows=call_rows, turn_updates=updates, replace=only_turn is None
    )
    return len(turn_rows)


def backfill(store: SessionStore, session_ids: list[str] | None = None) -> dict[str, int]:
    """Recompute every (or the given) session. Idempotent: derived rows are replaced."""
    return {sid: recompute_session(store, sid) for sid in session_ids or store.all_session_ids()}


# ---------------------------------------------------------------------- repair
def _common_prefix_cut(text: str, reference: str) -> tuple[int, int]:
    """Compare ignoring whitespace runs; return ``(cut index in text, matched non-space chars)``."""
    i = j = matched = 0
    cut = 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        while j < len(reference) and reference[j].isspace():
            j += 1
        if i >= len(text) or j >= len(reference) or text[i] != reference[j]:
            return cut, matched
        i, j, matched = i + 1, j + 1, matched + 1
        cut = i


def plan_attribution_repair(turns: list[dict[str, Any]], llm_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the moves that put interrupted responses back on their own turn.

    A move is found where a turn's ``bot_text`` starts with the previous turn's
    last LLM output (possibly cut short), followed by nothing or by the turn's
    own output. ``turns`` and ``llm_calls`` (with ``output_text``) are one session's rows.
    """
    outputs: dict[int, list[str]] = {}
    for call in sorted(llm_calls, key=lambda c: c["started_at"]):
        if (call.get("output_text") or "").strip():
            outputs.setdefault(call["turn_idx"], []).append(call["output_text"])
    by_idx = {turn["idx"]: turn for turn in turns}
    moves = []
    for idx in sorted(by_idx):
        turn, previous = by_idx[idx], by_idx.get(idx - 1)
        text = turn.get("bot_text") or ""
        if previous is None or not text or not outputs.get(idx - 1):
            continue
        reference = outputs[idx - 1][-1]
        cut, matched = _common_prefix_cut(text, reference)
        reference_chars = sum(not ch.isspace() for ch in reference)
        if matched < min(_REPAIR_MIN_CHARS, reference_chars):
            continue
        moved, rest = text[:cut].strip(), text[cut:].strip()
        if (previous.get("bot_text") or "").strip().endswith(moved):
            continue  # already on the previous turn (the bot repeated itself)
        own = " ".join(outputs.get(idx, []))
        if rest and _common_prefix_cut(rest, own)[1] < min(_REPAIR_MIN_CHARS, sum(not ch.isspace() for ch in rest)):
            continue  # the remainder is not this turn's own answer: leave the turn alone
        moves.append({"from_turn": idx, "to_turn": idx - 1, "text": moved, "remaining_text": rest})
    return moves


def repair_attribution(store: SessionStore, session_id: str, *, dry_run: bool = False) -> dict[str, Any] | None:
    """Move misattributed interrupted responses back one turn; returns the before/after report.

    Returns None when the session needs no repair: recorded by a fixed recorder,
    or already repaired (the audit annotation exists).
    """
    session = store.get_session(session_id)
    if session is None or ((session.get("config") or {}).get("recorder_version") or 1) >= RECORDER_VERSION:
        return None
    if store.annotations_for(session_id, kind=REPAIR_ANNOTATION):
        return None
    c, t = schema.llm_calls, schema.turns
    # Only columns that exist in every schema version: a dry run works before the migration.
    columns = (t.c.idx, t.c.bot_text, t.c.interrupted, t.c.bot_started_at, t.c.bot_stopped_at, t.c.user_started_at)
    with store.engine.connect() as conn:
        turns = [dict(r) for r in conn.execute(select(*columns).where(t.c.session_id == session_id)).mappings()]
        calls = [
            dict(r)
            for r in conn.execute(
                select(c.c.turn_idx, c.c.started_at, c.c.output_text).where(c.c.session_id == session_id)
            ).mappings()
        ]
    moves = plan_attribution_repair(turns, calls)
    keys = ("bot_text", "interrupted", "bot_stopped_at")
    after = {turn["idx"]: {k: turn.get(k) for k in keys} for turn in turns}
    before = {idx: dict(values) for idx, values in after.items()}
    started = {turn["idx"]: turn for turn in turns}
    last_idx = max(after, default=0)
    sources = {move["from_turn"] for move in moves}
    targets = {move["to_turn"] for move in moves}
    for move in moves:  # first take the foreign text out of every turn ...
        after[move["from_turn"]]["bot_text"] = move["remaining_text"]
    for move in moves:  # ... then give it to the turn it belongs to
        target = after[move["to_turn"]]
        target["bot_text"] = f"{target['bot_text'] or ''} {move['text']}".strip()
        target["interrupted"] = True
        # The response was cut when the next user turn started.
        target["bot_stopped_at"] = started[move["from_turn"]].get("user_started_at")
    for idx in sources - targets:
        values, turn = after[idx], started[idx]
        if idx != last_idx:
            values["interrupted"] = False  # the flag came with the moved response
        bot_started = turn.get("bot_started_at")
        if values["bot_stopped_at"] is not None and bot_started is not None and values["bot_stopped_at"] < bot_started:
            values["bot_stopped_at"] = None  # that was the previous response stopping
    changed = sorted(idx for idx in after if after[idx] != before[idx])
    report = {
        "session_id": session_id,
        "moves": moves,
        "turns": [{"idx": idx, "before": before[idx], "after": after[idx]} for idx in changed],
    }
    if not dry_run:
        store.write_derived(
            session_id,
            turn_rows=[],
            call_rows=[],
            turn_updates={idx: after[idx] for idx in changed},
            replace=False,
        )
        store.add_annotations(
            [
                {
                    "session_id": session_id,
                    "target_type": "session",
                    "target_id": session_id,
                    "source": "monitoring.turn_metrics",
                    "source_version": str(RECORDER_VERSION),
                    "kind": REPAIR_ANNOTATION,
                    "value": report,
                }
            ]
        )
    return report


def _print_repair(report: dict[str, Any]) -> None:
    def brief(values: dict[str, Any]) -> str:
        text = values["bot_text"]
        text = text if text is None or len(text) <= 70 else f"{text[:67]}..."
        flag, stopped = bool(values["interrupted"]), values["bot_stopped_at"]
        return f"interrupted={flag!s:<5} bot_stopped_at={stopped} bot_text={text!r}"

    print(f"session {report['session_id']}: {len(report['moves'])} response(s) moved")
    for turn in report["turns"]:
        print(f"  turn {turn['idx']:>3} before: {brief(turn['before'])}")
        print(f"           after:  {brief(turn['after'])}")


def main() -> int:
    """Entry point."""
    from dotenv import load_dotenv

    from monitoring.config import load_monitoring_config

    load_dotenv(override=False)
    parser = argparse.ArgumentParser(description="Backfill per-turn metrics and tool-call rows for recorded sessions.")
    parser.add_argument("--session", action="append", help="restrict to session id(s)")
    parser.add_argument(
        "--repair-attribution",
        action="store_true",
        help="first move interrupted responses stored on the following turn back to their own turn (older recordings)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="with --repair-attribution: print the plan, change nothing"
    )
    args = parser.parse_args()
    if args.dry_run and not args.repair_attribution:
        parser.error("--dry-run only applies to --repair-attribution")
    store = SessionStore(load_monitoring_config().db_url)
    if not args.dry_run:
        store.create_schema()
    if args.repair_attribution:
        repaired = 0
        for session_id in args.session or store.all_session_ids():
            report = repair_attribution(store, session_id, dry_run=args.dry_run)
            if report is not None and report["moves"]:
                repaired += 1
                _print_repair(report)
        print(f"{'Would repair' if args.dry_run else 'Repaired'} {repaired} session(s).")
        if args.dry_run:
            return 0
    result = backfill(store, args.session)
    print(f"Recomputed {sum(result.values())} turn(s) in {len(result)} session(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
