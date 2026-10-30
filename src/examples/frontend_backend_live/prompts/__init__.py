# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt sets for the cascade's own model calls and its delegations' input. A set is one module of plain text only.

``PROMPT_SETS`` lists every set by name: one set, ``default`` (``default.py``), which a configuration that names none
also gets. ``config.yaml`` chooses a set with ``prompt_version``. A new set is one module that builds its
``PROMPT_SET`` (see ``promptset.py``) and one entry here.
"""

from . import default
from .promptset import CallHints, NormalizerPrompt, PromptSet, TranscriptText, WorkTexts, WriterNotes

__all__ = [
    "DEFAULT_PROMPT_SET",
    "PROMPT_SETS",
    "CallHints",
    "NormalizerPrompt",
    "PromptSet",
    "TranscriptText",
    "WorkTexts",
    "WriterNotes",
    "get_prompt_set",
]

# The set a configuration gets when it names none.
DEFAULT_PROMPT_SET = "default"

# Every set by its name: the one set, default.
PROMPT_SETS = {"default": default.PROMPT_SET}


def get_prompt_set(name: str) -> PromptSet:
    """Return the prompt set called ``name`` (``DEFAULT_PROMPT_SET`` when empty).

    Raises:
        ValueError: No set has that name; the message lists the known ones.
    """
    try:
        return PROMPT_SETS[name or DEFAULT_PROMPT_SET]
    except KeyError:
        raise ValueError(f"Unknown prompt version {name!r}; known: {', '.join(PROMPT_SETS)}") from None
