# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Speech specific text cleaning filter."""

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

_TTS_RESERVED_CHARACTERS = re.compile(
    r"<(?=[A-Za-z/!])"  # < that starts a tag: <b>, </em>, <!--
    r"|[*{}]"  # Markdown asterisks and ARPAbet phoneme delimiters: *, {, }
)

# Magpie rejects a segment that contains only punctuation ("Invalid text, only
# punctuation"), and that error ends the whole synthesis stream for the reply.
# Drop lines with no letters or digits (for example a lone "?" left after a
# sentence split, or a "---" separator) and punctuation-only prefixes before a
# word, such as list bullets ("- ", "• ").
_PUNCTUATION_ONLY_SEGMENT = re.compile(r"(?m)^[^\w\n]*$\n?|^\s*[^\w\s]+\s+(?=\w)")


def _clean_for_nvidia_tts(text: str) -> str:
    """Remove reserved characters and punctuation-only segments from TTS input."""
    text = _TTS_RESERVED_CHARACTERS.sub("", text)
    return _PUNCTUATION_ONLY_SEGMENT.sub("", text)


class NemotronSpeechTextFilter(BaseTextFilter):
    """Strips characters reserved by the NVIDIA TTS text preprocessor.

    ``{...}``  ARPAbet phoneme notation.

    ``<tag>``  SSML tags.

    ``*``  Markdown emphasis markers.

    It also drops punctuation-only lines and prefixes, which Magpie rejects.
    """

    async def filter(self, text: str) -> str:
        """Strip reserved characters and punctuation-only segments from TTS input."""
        return _clean_for_nvidia_tts(text)


class NemotronSpeechMarkdownTextFilter(MarkdownTextFilter):
    """Markdown filter safe for NVIDIA TTS.

    Extends Pipecat's :class:`MarkdownTextFilter` with a final pass that strips
    characters reserved by the NVIDIA TTS preprocessor.  Use this instead of
    ``MarkdownTextFilter`` wherever the output feeds into NVIDIA TTS
    service.
    """

    async def filter(self, text: str) -> str:
        """Apply Markdown stripping, then remove NVIDIA TTS reserved characters and punctuation-only segments."""
        text = await super().filter(text)
        return _clean_for_nvidia_tts(text)
