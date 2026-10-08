# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A conservative structural floor against replaying stale voice answers."""

import re
from difflib import SequenceMatcher

from examples.frontend_backend_live.common.history import BACKEND_RESULT_PREFIX


def normalized(text):
    """Return ``text`` as lowercase words, for comparison."""
    return " ".join(re.findall(r"\w+", text.lower()))


def repeated_speech(speech, caller, history):
    """Return how closely ``speech`` repeats a recent reply (0.0 when it does not, or the caller asked to repeat)."""
    if len(speech.split()) < 6 or re.search(r"\brepeat\b|say that again|didn['’]?t catch", caller, re.I):
        return 0.0
    candidates = [
        item["content"]
        for item in history
        if item["role"] == "assistant" or str(item["content"]).startswith(BACKEND_RESULT_PREFIX)
    ][-6:]
    ratios = [
        SequenceMatcher(
            None,
            normalized(speech),
            normalized(text.removeprefix(BACKEND_RESULT_PREFIX)),
        ).ratio()
        for text in candidates
    ]
    similarity = max(ratios, default=0.0)
    return similarity if similarity >= 0.85 else 0.0
