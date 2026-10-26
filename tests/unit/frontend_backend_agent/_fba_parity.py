# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G2 decision-level comparison of a Pipecat-adapter run with v1's golden capture.

Plan ``misc/frontend-backend-verdict/shared-stack-plan.md`` §10.1 G2. Both captures are already
G0-normalized (``_fba_golden.run_scenario``). This module turns each into a :class:`Decisions`
view that keeps what must be identical and sets aside what the behaviour register lets differ:

* events are split by delivery class (urgent / stream / response, B10); order is compared
  within each class only, and ids are renumbered by order of appearance within that view;
* ``response.output_audio.delta`` events are dropped and audio is compared per item by total
  duration (B10: Pipecat sends 10 ms chunks, v1 sent 100 ms deltas);
* audio-clock fields (``audio_start_ms``, ``audio_end_ms``, ``heard_ms`` and the other
  :data:`TIMING_KEYS`) are pulled out and compared within the B1 windows by :func:`compare`;
  ``audio_ms`` (the input clock when a record was written) is a timestamp and is ignored;
* JSONL records are compared the same way, without clock fields and without the fields the
  adapter adds to v1's records (S21: ``adapter``, ``pipecat``, ``onset_source``; §7.3: ``prompts``;
  ``session_updated``'s ``domain`` and ``domain_source``);
* ``conversation.item.input_audio_transcription.delta`` events are compared per item by text,
  not by position: v1's stub recognizer reports its one interim when the utterance is
  finalized (after the commit), while Pipecat's STT reports interims before the VAD stop, as
  a real recognizer does in both versions.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

#: Server events the adapter sends at once, ahead of queued agent audio (B10).
URGENT_PREFIXES = ("session.", "input_audio_buffer.", "conversation.item.input_audio_transcription.")
URGENT_TYPES = frozenset({"error", "conversation.item.truncated", "conversation.item.deleted"})
STREAM_TYPES = frozenset({"response.output_audio_transcript.delta", "response.output_text.delta"})
AUDIO_DELTA = "response.output_audio.delta"
#: Audio-clock fields compared within :data:`WINDOW_MS`, not exactly (B1, B4, B9).
TIMING_KEYS = frozenset(
    {"audio_start_ms", "audio_end_ms", "heard_ms", "played_ms", "sent_ms", "heard_at_speech_ms", "summary_end_ms"}
)
#: The B1 window: Pipecat's VAD confirms a start after ``start_secs`` (0.2 s) and measures the
#: stop with Silero instead of v1's energy segmenter; every audio-clock field may move by it.
WINDOW_MS = 200.0
#: Input-clock timestamps of log records: ignored (G2 ignores timestamps).
TIMESTAMP_KEYS = frozenset({"audio_ms"})
#: Pipecat's transport pads each sentence's last chunk to a full 20 ms chunk (``out_10ms_chunks: 2``).
PAD_MS_PER_SENTENCE = 20.0
#: Record fields the adapter adds to v1's (S21, and ``session_updated``'s service domain).
ADDED_RECORD_KEYS = frozenset({"adapter", "pipecat", "onset_source", "prompts", "domain", "domain_source"})
INTERIM_DELTA = "conversation.item.input_audio_transcription.delta"
_PLACEHOLDER = re.compile(r"^<([A-Z_]+?)(\d+)>$")


def event_class(event: dict[str, Any]) -> str:
    """``urgent``, ``stream``, ``response`` or ``audio`` (B10's delivery classes)."""
    kind = str(event.get("type", ""))
    if kind == AUDIO_DELTA:
        return "audio"
    if kind in STREAM_TYPES:
        return "stream"
    if kind in URGENT_TYPES or kind.startswith(URGENT_PREFIXES):
        return "urgent"
    if kind in ("conversation.item.added", "conversation.item.done"):
        item = event.get("item") or {}
        if item.get("role") == "user" or item.get("type") == "function_call_output":
            return "urgent"
    return "response"


@dataclass
class Decisions:
    """One capture, split for comparison."""

    urgent: list[dict[str, Any]] = field(default_factory=list)
    stream: list[dict[str, Any]] = field(default_factory=list)
    response: list[dict[str, Any]] = field(default_factory=list)
    audio_ms: dict[str, float] = field(default_factory=dict)
    interims: dict[str, str] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[tuple[str, str], float] = field(default_factory=dict)  # (where, key) -> value
    sentences: dict[str, int] = field(default_factory=dict)  # transcript deltas per item


def _audio_ms(event: dict[str, Any], rate: int, width: int) -> float:
    data = base64.b64decode(event.get("delta", "") or "")
    return len(data) / width / rate * 1000.0


def _strip_timing(value: Any, where: str, out: dict[tuple[str, str], float]) -> Any:
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if key in TIMESTAMP_KEYS:
                continue
            if key in TIMING_KEYS and isinstance(item, (int, float)) and not isinstance(item, bool):
                out[(where, key)] = float(item)
                continue
            kept[key] = _strip_timing(item, where, out)
        return kept
    if isinstance(value, list):
        return [_strip_timing(item, where, out) for item in value]
    return value


class _Renumber:
    """Renumber G0 placeholders (``<ITEM3>``) by order of appearance within one view."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def __call__(self, value: Any) -> Any:
        if isinstance(value, dict):  # sorted: the fixtures were written with sorted keys
            return {key: self(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [self(item) for item in value]
        if isinstance(value, str):
            match = _PLACEHOLDER.match(value)
            if match:
                if value not in self.mapping:
                    label = match.group(1)
                    self.counters[label] = self.counters.get(label, 0) + 1
                    self.mapping[value] = f"<{label}{self.counters[label]}>"
                return self.mapping[value]
        return value


def decisions(capture: dict[str, Any], *, rate: int = 8000, width: int = 1) -> Decisions:
    """Split a normalized capture (``events``, ``records``) into its comparable parts.

    ``rate`` and ``width`` describe the output audio format (benchmark client: 8 kHz mu-law, one byte).
    """
    out = Decisions()
    renumber = _Renumber()
    classes: dict[str, list[dict[str, Any]]] = {"urgent": [], "stream": [], "response": []}
    audio: list[dict[str, Any]] = []
    interims: list[dict[str, Any]] = []
    for event in capture["events"]:
        cls = event_class(event)
        if cls == "audio":
            audio.append(event)
            continue
        if event.get("type") == INTERIM_DELTA:
            interims.append(event)
            continue
        where = f"{cls}[{len(classes[cls])}] {event.get('type')}"
        classes[cls].append(_strip_timing(event, where, out.timing))
    # Renumber in a fixed class order, so a reordering across classes does not change ids.
    out.urgent = renumber(classes["urgent"])
    out.response = renumber(classes["response"])
    out.stream = renumber(classes["stream"])
    for event in audio:
        item = renumber(event.get("item_id", ""))
        out.audio_ms[item] = out.audio_ms.get(item, 0.0) + _audio_ms(event, rate, width)
    for event in out.stream:
        item = str(event.get("item_id", ""))
        out.sentences[item] = out.sentences.get(item, 0) + 1
    for event in interims:
        item = renumber(event.get("item_id", ""))
        out.interims[item] = out.interims.get(item, "") + str(event.get("delta", ""))
    records = [
        _strip_timing(
            {key: value for key, value in record.items() if key not in ADDED_RECORD_KEYS},
            f"record[{index}] {record.get('kind')}",
            out.timing,
        )
        for index, record in enumerate(capture["records"])
    ]
    out.records = renumber(records)
    return out


@dataclass(frozen=True)
class Allowances:
    """What one scenario may show beyond the general tolerances (each case named in the test).

    ``cut_items``: items cut by a barge-in or cancel; where the cut lands moves with the B1/B9
    windows, so their audio is only checked to be no longer than the uncut item would be.
    ``heard_text``: record kinds whose ``heard`` text may differ (both must be prefixes of the
    item's ``full_text``), because the playout cursor at the barge-in moves with the windows.
    """

    cut_items: frozenset[str] = frozenset()
    heard_text: frozenset[str] = frozenset()


def _without_heard(records: list[dict[str, Any]], kinds: frozenset[str]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in record.items() if key != "heard"} if record.get("kind") in kinds else record
        for record in records
    ]


def compare(want: Decisions, got: Decisions, allow: Allowances | None = None) -> list[str]:
    """Every G2 difference between v1 (``want``) and v2 (``got``) beyond the allowances."""
    allow = allow or Allowances()
    problems: list[str] = []
    for part in ("urgent", "stream", "response"):
        if getattr(want, part) != getattr(got, part):
            problems.append(f"{part} events differ")
    if want.interims != got.interims:
        problems.append(f"interim transcripts differ: {want.interims} != {got.interims}")
    if _without_heard(want.records, allow.heard_text) != _without_heard(got.records, allow.heard_text):
        problems.append("records differ")
    for record_want, record_got in zip(want.records, got.records, strict=False):
        if record_want.get("kind") in allow.heard_text and "heard" in record_want:
            full = str(record_want.get("full_text", ""))
            for heard in (str(record_want["heard"]), str(record_got.get("heard", ""))):
                if not full.startswith(heard):
                    problems.append(f"heard text {heard!r} is not a prefix of {full!r}")
    if want.timing.keys() != got.timing.keys():
        problems.append(f"timing fields differ: {sorted(set(want.timing) ^ set(got.timing))}")
    for key in want.timing.keys() & got.timing.keys():
        if abs(want.timing[key] - got.timing[key]) > WINDOW_MS:
            problems.append(f"{key}: {got.timing[key]} vs v1 {want.timing[key]} (window {WINDOW_MS} ms)")
    if want.audio_ms.keys() != got.audio_ms.keys():
        problems.append(f"audio items differ: {sorted(want.audio_ms)} != {sorted(got.audio_ms)}")
    for item in want.audio_ms.keys() & got.audio_ms.keys():
        if item in allow.cut_items:
            continue
        pad = PAD_MS_PER_SENTENCE * max(1, want.sentences.get(item, 1))
        delta = got.audio_ms[item] - want.audio_ms[item]
        if not -1.0 <= delta <= pad:
            problems.append(f"audio of {item}: {got.audio_ms[item]:.1f} ms vs v1 {want.audio_ms[item]:.1f} ms")
    return problems
