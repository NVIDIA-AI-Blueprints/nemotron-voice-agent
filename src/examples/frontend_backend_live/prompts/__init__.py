# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt sets for the cascade's own model calls. A set is one module, chosen by name, and is plain text only.

``PROMPT_SETS`` lists every set by name; ``config.yaml`` chooses one with ``prompt_version``. A new set is one module
that builds its ``PROMPT_SET`` (see ``promptset.py``) and one entry here.
"""

from . import v1, v2, v3
from .promptset import CallHints, NormalizerPrompt, PromptSet, WriterNotes

__all__ = [
    "DEFAULT_PROMPT_SET",
    "PROMPT_SETS",
    "CallHints",
    "NormalizerPrompt",
    "PromptSet",
    "WriterNotes",
    "get_prompt_set",
]

# The set a configuration gets when it names none.
DEFAULT_PROMPT_SET = "v2"

# v3 is listed for measurement; no shipped configuration names it.
PROMPT_SETS = {prompts.name: prompts for prompts in (v1.PROMPT_SET, v2.PROMPT_SET, v3.PROMPT_SET)}


def get_prompt_set(name: str) -> PromptSet:
    """Return the prompt set called ``name`` (``DEFAULT_PROMPT_SET`` when empty).

    Raises:
        ValueError: No set has that name; the message lists the known ones.
    """
    try:
        return PROMPT_SETS[name or DEFAULT_PROMPT_SET]
    except KeyError:
        raise ValueError(f"Unknown prompt version {name!r}; known: {', '.join(PROMPT_SETS)}") from None
