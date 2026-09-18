# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only state helpers for cumulative streaming ASR transcripts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def normalize_transcript(text: str) -> str:
    """Normalize ASR whitespace while preserving recognized words."""
    return " ".join((text or "").split())


def extract_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer list or mapping output to one token-id list."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Tokenizer output did not contain an input_ids sequence")
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValueError("Only one prompt sequence is supported")
        value = value[0]
    return [int(token_id) for token_id in value]


@dataclass(frozen=True, slots=True)
class TranscriptCommit:
    """Stable append-only part of one cumulative ASR hypothesis."""

    observed: str
    committed: str
    text_delta: str
    reset_required: bool


class StableTranscriptCommitter:
    """Hold the mutable ASR tail and surface committed-prefix revisions."""

    def __init__(self, *, hold_words: int = 1) -> None:
        """Create a committer that keeps ``hold_words`` provisional."""
        if hold_words < 0:
            raise ValueError("hold_words must be >= 0")
        self.hold_words = hold_words
        self.committed = ""

    def observe(self, transcript: str, *, is_final: bool = False) -> TranscriptCommit:
        """Commit the stable prefix from a cumulative ASR update."""
        observed = normalize_transcript(transcript)
        words = observed.split()
        held = 0 if is_final else min(self.hold_words, len(words))
        target = " ".join(words if held == 0 else words[:-held])

        extends_committed = not self.committed or target == self.committed or target.startswith(f"{self.committed} ")
        reset_required = bool(self.committed and target and not extends_committed)
        if reset_required:
            text_delta = target
        elif self.committed:
            text_delta = target[len(self.committed) :]
        else:
            text_delta = target

        self.committed = target
        return TranscriptCommit(
            observed=observed,
            committed=target,
            text_delta=text_delta,
            reset_required=reset_required,
        )
