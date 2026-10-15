# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the client has received, and what it has heard (plan section 8.3).

Three quantities are tracked separately on one output timeline (a client such
as tau2 plays every agent item back to back from a single buffer):

* ``generating``: the active response is still producing or sending audio;
* ``sent_ms`` plus per-sentence spans: audio delivered to the client so far;
* ``heard_ms``: the client's playout cursor. Each input-audio step of ``dt``
  advances it by ``dt``, capped by ``sent_ms`` *at every step*, which models a
  client that plays 1x in lock step with the audio it streams and pauses when
  its buffer runs dry. A TTS stall therefore never inflates ``heard_ms``.

``active = generating or heard_ms < sent_ms``: a response can be interrupted
even when nothing unplayed is buffered (TTS still synthesizing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class SentenceSpan:
    """One sentence of an item on the output timeline."""

    text: str
    start_ms: float
    end_ms: float


@dataclass(slots=True)
class ItemSpan:
    """One assistant message item on the output timeline."""

    item_id: str
    kind: str
    full_text: str
    start_ms: float
    sentences: list[SentenceSpan] = field(default_factory=list)
    repaired: bool = False

    @property
    def end_ms(self) -> float:
        """End of the item's delivered audio."""
        return self.sentences[-1].end_ms if self.sentences else self.start_ms


def heard_prefix(sentences: list[SentenceSpan], heard_ms: float) -> str:
    """Text heard by ``heard_ms``: whole sentences, then a proportional, word-snapped prefix."""
    heard: list[str] = []
    for sentence in sentences:
        if sentence.end_ms <= sentence.start_ms:
            break  # no audio delivered for this sentence yet: nothing from here on was heard
        if sentence.end_ms <= heard_ms:
            heard.append(sentence.text)
            continue
        if heard_ms > sentence.start_ms and sentence.end_ms > sentence.start_ms:
            fraction = (heard_ms - sentence.start_ms) / (sentence.end_ms - sentence.start_ms)
            count = math.ceil(fraction * len(sentence.text))
            partial = sentence.text[:count]
            if count < len(sentence.text) and not sentence.text[count : count + 1].isspace():
                cut = partial.rfind(" ")
                partial = partial[:cut] if cut >= 0 else ""
            if partial.strip():
                heard.append(partial.rstrip())
        break
    return " ".join(part.strip() for part in heard if part.strip())


class ResponseProgress:
    """Output timeline for one session."""

    def __init__(self) -> None:
        """Start with nothing sent."""
        self.generating = False
        self.sent_ms = 0.0
        self.heard_ms = 0.0
        self.items: list[ItemSpan] = []
        self._open_sentence: SentenceSpan | None = None
        self.first_audio_sent = False

    @property
    def active(self) -> bool:
        """Whether a barge-in has anything to interrupt."""
        return self.generating or self.heard_ms < self.sent_ms

    @property
    def unplayed_ms(self) -> float:
        """Audio delivered but not yet heard."""
        return max(0.0, self.sent_ms - self.heard_ms)

    def begin_item(self, item_id: str, *, kind: str, full_text: str) -> ItemSpan:
        """Start an item at the current end of the timeline."""
        span = ItemSpan(item_id=item_id, kind=kind, full_text=full_text, start_ms=self.sent_ms)
        self.items.append(span)
        return span

    def begin_sentence(self, text: str) -> None:
        """Start a sentence of the newest item."""
        sentence = SentenceSpan(text=text, start_ms=self.sent_ms, end_ms=self.sent_ms)
        self.items[-1].sentences.append(sentence)
        self._open_sentence = sentence

    def add_audio(self, ms: float) -> None:
        """Record ``ms`` of audio delivered for the open sentence."""
        self.sent_ms += ms
        if self._open_sentence is not None:
            self._open_sentence.end_ms = self.sent_ms
        self.first_audio_sent = True

    def end_sentence(self) -> None:
        """Close the open sentence."""
        self._open_sentence = None

    def advance(self, step_ms: float) -> None:
        """Input audio advanced by ``step_ms``: move the playout cursor, capped by delivered audio."""
        self.heard_ms = min(self.heard_ms + step_ms, self.sent_ms)

    def item(self, item_id: str) -> ItemSpan | None:
        """Look up an item span."""
        for span in reversed(self.items):
            if span.item_id == item_id:
                return span
        return None

    def heard_text(self, item_id: str) -> str:
        """Text of ``item_id`` heard so far."""
        span = self.item(item_id)
        return heard_prefix(span.sentences, self.heard_ms) if span else ""

    def item_heard_ms(self, item_id: str) -> float:
        """Milliseconds of ``item_id`` heard so far."""
        span = self.item(item_id)
        if span is None:
            return 0.0
        return max(0.0, min(self.heard_ms, span.end_ms) - span.start_ms)

    def discard_unplayed(self) -> None:
        """After a barge-in the client dropped its buffer: the timeline ends at the cursor."""
        cursor = self.heard_ms
        for span in self.items:
            span.sentences = [s for s in span.sentences if s.start_ms < cursor or s.end_ms <= cursor]
            for sentence in span.sentences:
                if sentence.end_ms > cursor:
                    # Keep the heard fraction of a cut sentence: shrink its text with its span.
                    sentence.text = heard_prefix([sentence], cursor)
                    sentence.end_ms = cursor
        self.sent_ms = cursor
        self._open_sentence = None
