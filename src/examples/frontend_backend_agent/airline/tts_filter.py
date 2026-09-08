# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Frontend/Backend Agent-specific TTS text cleanup."""

from __future__ import annotations

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

from examples.frontend_backend_agent.airline.airports import AIRPORT_SPOKEN_NAMES, airport_spoken_name

_PNR_ACRONYM_RE = re.compile(r"\bPNR(?:(['’])s)?\b")
_PNR_CODE_RE = re.compile(r"\b([A-Z]{3})(\d{3})\b")
_FLIGHT_RE = re.compile(r"\b([A-Z]{2})(\d{2,4})\b")
_AIRPORT_RE = re.compile(r"\b(" + "|".join(re.escape(code) for code in sorted(AIRPORT_SPOKEN_NAMES)) + r")\b")
_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:-\s+|\d+\.\s+)")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")


class FrontendBackendAgentPronunciationTextFilter(BaseTextFilter):
    """Apply flight-domain pronunciation substitutions after generic cleanup."""

    async def filter(self, text: str) -> str:
        """Return TTS input with airline codes expanded for clearer speech."""
        text = _normalize_spacing(text)
        text = _LIST_PREFIX_RE.sub("", text)
        text = _PNR_ACRONYM_RE.sub(_sub_pnr_acronym, text)
        text = _PNR_CODE_RE.sub(_spell_groups, text)
        text = _FLIGHT_RE.sub(_spell_groups, text)
        text = _AIRPORT_RE.sub(lambda match: airport_spoken_name(match.group(1)), text)
        return _normalize_spacing(text)


async def apply_frontend_backend_agent_pronunciation_for_tts(text: str, _aggregation_type: object) -> str:
    """Apply Frontend/Backend Agent pronunciation substitutions only to TTS synthesis text."""
    return await FrontendBackendAgentPronunciationTextFilter().filter(text)


def _sub_pnr_acronym(match: re.Match[str]) -> str:
    if match.group(1):
        return "P N R code"
    return "P N R"


def _spell_groups(match: re.Match[str]) -> str:
    return " ".join(_spell_chars(group) for group in match.groups())


def _spell_chars(value: str) -> str:
    return " ".join(value)


def _normalize_spacing(text: str) -> str:
    text = text.replace("\u202f", " ").replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return _BLANK_LINE_RE.sub("\n\n", text).strip()
