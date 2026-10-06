# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Keep a backend answer the caller did not hear from being lost to their speech.

One session-local record follows the latest answer, question or consent
question the backend handed to the Talker:

* **held**: it arrived while the caller was speaking, so no reply has spoken it
  yet. When the caller's next words only acknowledge or ask for progress
  ("okay", "any update?"), the Talker is told to give it now.
* **cut**: a reply that was speaking it, or about to, was cancelled by caller
  speech. If those words only acknowledge or ask for progress, the Talker is
  told to say it again, once. A consent question is asked again in full and is
  never resumed mid-sentence; a partly heard question approves nothing.

Substantive words (a new request, a correction, a withdrawal) drop the record:
the Talker plans from them as before. Fillers never enter the record, so they
are never repeated.

Before this, the Talker's only signal was the earlier tool result in its
context. Repeating it was rejected as a cached replay, and the Talker sent the
same request to the backend again instead of speaking the waiting answer.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from loguru import logger
from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame, InterruptionFrame

from examples.frontend_backend_agent.src.frontend_verdict import is_progress_phrase, pure_ack_or_progress

ItemKind = Literal["answer", "question", "consent"]
ItemState = Literal["held", "started", "speaking", "heard", "cut", "done"]

#: Payload reasons that are status notices rather than something the caller is owed.
_UNTRACKED_REASONS = frozenset(
    {
        "aborted",
        "cancelled",
        "interrupted_speech",
        "nothing_to_cancel",
        "no_action_needed",
        "planner_error",
        "planner_error_exhausted",
        "timeout",
        "tool_error",
    }
)
_QUESTION_REASONS = frozenset({"params_missing", "params_invalid"})

ANSWER_REDELIVERY_NOTE = (
    "The caller did not hear your reply to the latest call_backend result: their speech cut it off. "
    "Their latest words only acknowledged you or asked for progress. Give that answer now, in full, "
    "grounded only in that result. Do not call any function and do not mention that you were cut off."
)
HELD_ANSWER_NOTE = (
    "The latest call_backend result arrived while the caller was speaking, so they have not heard it yet. "
    "Their latest words only acknowledged you or asked for progress. Give that answer now, grounded only in "
    "that result. Do not call any function."
)
QUESTION_REDELIVERY_NOTE = (
    "The caller did not hear your last question in full; their latest words only acknowledged you or asked "
    "for progress. Ask it again now, exactly as written, and nothing else. Do not call any function. "
    "The following JSON string is the question: {text}"
)
CONSENT_REDELIVERY_NOTE = (
    "The caller did not hear your consent question in full, so nothing is approved. Ask it again now, in full "
    "and exactly as written, from its first word; do not resume mid-sentence and do not call any function. "
    "The following JSON string is the question: {text}"
)


@dataclass(slots=True)
class _Item:
    """The latest backend payload handed to the Talker and what became of it."""

    item_id: int
    kind: ItemKind
    text: str
    context: str
    run_id: str | None
    state: ItemState = "held"
    audio_sent: bool = False
    redelivered: bool = False


@dataclass(frozen=True, slots=True)
class TalkerRunPlan:
    """What one Talker run must add for an unheard answer, if anything."""

    note: str | None = None
    #: The run repeats an earlier backend answer on purpose; it is not a cached replay.
    allow_replay: bool = False


NO_PLAN = TalkerRunPlan()


class DeliveryTracker:
    """Follow one session's latest backend answer from hand-off to the caller's ears."""

    def __init__(self, *, hold_answers: bool = True, redeliver_cut_answers: bool = True) -> None:
        """Create the tracker with the two switches it serves."""
        self._hold_answers = hold_answers
        self._redeliver_cut = redeliver_cut_answers
        self._item: _Item | None = None
        self._ids = itertools.count(1)

    @property
    def enabled(self) -> bool:
        """Return whether either behaviour is on."""
        return self._hold_answers or self._redeliver_cut

    @property
    def state(self) -> str | None:
        """Return the tracked item's state, for tests and diagnostics."""
        return self._item.state if self._item is not None else None

    def note_payload(self, payload: Mapping[str, object]) -> None:
        """Start tracking a speakable backend payload just handed to the Talker."""
        if not self.enabled:
            return
        kind = _payload_kind(payload)
        if kind is None:
            return
        previous = self._item
        if previous is not None and previous.state in {"held", "cut"}:
            self._log("discarded", previous, reason="replaced")
        self._item = _Item(
            item_id=next(self._ids),
            kind=kind,
            text=" ".join(str(payload.get("response_text") or "").split()),
            context=str(payload.get("context") or payload.get("tool") or ""),
            run_id=_optional_str(payload.get("run_id")),
        )

    def begin_talker_run(self, latest_role: str | None, latest_user_text: str) -> TalkerRunPlan:
        """Decide what a starting Talker run must do for the tracked item.

        ``latest_role`` is the role of the newest conversation message. A run
        that answers a tool result is the ordinary delivery. A run that answers
        a user turn decides the fate of a held or cut item from those words.
        """
        item = self._item
        if item is None or item.state not in {"held", "cut"}:
            return NO_PLAN
        if latest_role != "user":
            item.state = "started"
            return NO_PLAN
        acknowledgement = pure_ack_or_progress(latest_user_text) or is_progress_phrase(latest_user_text)
        if item.state == "held":
            if not acknowledgement or not self._hold_answers:
                # The Talker sees the result beside the new words and decides; this is today's path.
                self._log("held_answer", item, outcome="left_to_talker")
                item.state = "started"
                return NO_PLAN
            self._log("held_answer", item, outcome="delivered_after_acknowledgement")
            item.state = "started"
            return TalkerRunPlan(note=_note(item, held=True), allow_replay=True)
        if not acknowledgement:
            self._log("discarded", item, reason="substantive")
            item.state = "done"
            return NO_PLAN
        if item.redelivered or not self._redeliver_cut:
            self._log("discarded", item, reason="already_redelivered" if item.redelivered else "disabled")
            item.state = "done"
            return NO_PLAN
        item.redelivered = True
        item.state = "started"
        self._log("redelivered", item)
        return TalkerRunPlan(note=_note(item, held=False), allow_replay=True)

    def talker_output(self, *, spoke: bool, called_function: bool) -> None:
        """Record what the Talker run for the tracked item produced."""
        item = self._item
        if item is None or item.state != "started":
            return
        if called_function and not spoke:
            # The Talker delegated again; a newer payload will replace this one.
            self._log("discarded", item, reason="talker_redelegated")
            item.state = "done"
        elif spoke:
            item.state = "speaking"

    def observe(self, frame: Frame) -> None:
        """Follow audio and caller interruptions for the tracked item."""
        item = self._item
        if item is None:
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            if item.state in {"started", "speaking"}:
                item.state = "speaking"
                item.audio_sent = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if item.state == "speaking" and item.audio_sent:
                item.state = "heard"
        elif isinstance(frame, InterruptionFrame) and item.state in {"started", "speaking"}:
            item.state = "cut"
            self._log("cut", item)

    def _log(self, event: str, item: _Item, **fields: object) -> None:
        logger.bind(
            event=event,
            item_id=item.item_id,
            kind=item.kind,
            context=item.context,
            run_id=item.run_id,
            audio_sent=item.audio_sent,
            **fields,
        ).info(
            f"Backend answer {event}: item={item.item_id} kind={item.kind} audio_sent={item.audio_sent}"
            + "".join(f" {key}={value}" for key, value in fields.items())
        )


def _payload_kind(payload: Mapping[str, object]) -> ItemKind | None:
    if payload.get("speakable") is False or not str(payload.get("response_text") or "").strip():
        return None
    reason = str(payload.get("reason") or "")
    if reason in _UNTRACKED_REASONS:
        return None
    if reason == "confirmation_needed":
        return "consent"
    if payload.get("type") == "response_hint" and reason in _QUESTION_REASONS:
        return "question"
    return "answer"


def _note(item: _Item, *, held: bool) -> str:
    if item.kind == "consent" and item.text:
        return CONSENT_REDELIVERY_NOTE.format(text=json.dumps(item.text, ensure_ascii=False))
    if item.kind == "question" and item.text:
        return QUESTION_REDELIVERY_NOTE.format(text=json.dumps(item.text, ensure_ascii=False))
    return HELD_ANSWER_NOTE if held else ANSWER_REDELIVERY_NOTE


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None
