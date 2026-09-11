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


class NemotronSpeechTextFilter(BaseTextFilter):
    """Strips characters reserved by the NVIDIA TTS text preprocessor.

    ``{...}``  ARPAbet phoneme notation.

    ``<tag>``  SSML tags.

    ``*``  Markdown emphasis markers.
    """

    async def filter(self, text: str) -> str:
        """Strip markdown asterisks, SSML tag openers, and ARPAbet delimiters from TTS input."""
        return _TTS_RESERVED_CHARACTERS.sub("", text)


class NemotronSpeechMarkdownTextFilter(MarkdownTextFilter):
    """Markdown filter safe for NVIDIA TTS.

    Extends Pipecat's :class:`MarkdownTextFilter` with a final pass that strips
    characters reserved by the NVIDIA TTS preprocessor.  Use this instead of
    ``MarkdownTextFilter`` wherever the output feeds into NVIDIA TTS
    service.
    """

    async def filter(self, text: str) -> str:
        """Apply Markdown stripping then remove NVIDIA TTS reserved characters."""
        text = await super().filter(text)
        return _TTS_RESERVED_CHARACTERS.sub("", text)
