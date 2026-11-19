# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nemotron Speech specific text transforms."""

import re

from pipecat.utils.text.transforms import strip_markdown

_TTS_RESERVED_CHARACTERS = re.compile(
    r"<(?=[A-Za-z/!])"  # < that starts a tag: <b>, </em>, <!--
    r"|[*{}]"  # Markdown asterisks and ARPAbet phoneme delimiters: *, {, }
)


async def nemotron_speech_text_transform(text: str, _text_type: object) -> str:
    """Strip characters reserved by the NVIDIA TTS text preprocessor.

    ``{...}``  ARPAbet phoneme notation.

    ``<tag>``  SSML tags.

    ``*``  Markdown emphasis markers.
    """
    return _TTS_RESERVED_CHARACTERS.sub("", text)


async def nemotron_speech_markdown_text_transform(text: str, text_type: object) -> str:
    """Strip Markdown, then remove characters reserved by NVIDIA TTS."""
    stripped = await strip_markdown(text, text_type)
    return await nemotron_speech_text_transform(stripped, text_type)
