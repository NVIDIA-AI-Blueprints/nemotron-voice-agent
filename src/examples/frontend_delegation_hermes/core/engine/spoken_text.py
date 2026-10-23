# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer cleanup before TTS (``output.clean_answers``, tau3-failure-fixes-plan.md section 6).

Markdown is read aloud badly ("asterisk asterisk"), and a list read as one run-on line
merges items. :func:`clean_for_speech`:

* strips emphasis markers (``**``, ``__``, ``*``), heading marks, code ticks and URLs;
* turns every list item into its own sentence, keeping the item's values in it, so no
  item is merged with its neighbour's price or quantity;
* never touches a single underscore, so identifiers such as ``jordan_lee_82`` survive.

Pure; the turn manager applies it before the answer enters the transcript, so the
transcript records what was spoken.
"""

from __future__ import annotations

import re

_ITEM = re.compile(r"^\s*(?:[-*•+]|\d{1,2}[.)])\s+")
_HEADING = re.compile(r"^\s*#{1,6}\s*")
_BOLD = re.compile(r"(\*\*|__)(\S(?:.*?\S)?)\1")
_STAR = re.compile(r"(?<![\w*])\*(\S(?:[^*]*?\S)?)\*(?![\w*])")
_URL = re.compile(r"\(?\bhttps?://\S+?\)?(?=[\s,;]|[.!?](?:\s|$)|$)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_SPACES = re.compile(r"\s+")
_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:])")
_END = (".", "!", "?")


def clean_for_speech(text: str) -> str:
    """``text`` as plain spoken sentences (see the module docstring)."""
    sentences: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or set(line) <= set("-*_=|: "):
            continue  # blank lines, rules and table separators
        heading = bool(_HEADING.match(line))
        item = bool(_ITEM.match(line))
        line = _ITEM.sub("", _HEADING.sub("", line))
        line = _LINK.sub(r"\1", line)
        line = _URL.sub("", line)
        line = _BOLD.sub(r"\2", line)
        line = _STAR.sub(r"\1", line)
        line = line.replace("`", "").replace("|", ", ")
        line = _BEFORE_PUNCT.sub(r"\1", _SPACES.sub(" ", line)).strip(" ,")
        if not line:
            continue
        if (item or heading) and line.endswith((":", ";", ",")):
            line = line[:-1].rstrip()
        if (item or heading) and not line.endswith(_END):
            line = f"{line}."
        elif line.endswith(":") and not item:
            line = f"{line[:-1].rstrip()}."
        sentences.append(line)
    return " ".join(sentences)
