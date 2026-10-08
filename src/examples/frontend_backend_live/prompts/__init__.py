# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt sets for the bridge's own model calls. A set is selected by name and is plain text only."""

from dataclasses import dataclass

from . import v1, v2


@dataclass(frozen=True)
class PromptSet:
    """The text a prompt version contributes to each model call."""

    name: str
    routing: str  # appended to the client's instructions for the routing decision
    commentary: str  # appended for commentary phrasing
    answer: str  # appended when the update is a waiting caller's answer
    backend_notes: str  # appended to the backend's own instructions ("" for none)


PROMPT_SETS = {
    module.__name__.rsplit(".", 1)[-1]: PromptSet(
        module.__name__.rsplit(".", 1)[-1],
        module.ROUTING_INSTRUCTIONS,
        module.COMMENTARY_INSTRUCTIONS,
        module.ANSWER_INSTRUCTIONS,
        module.BACKEND_VOICE_NOTES,
    )
    for module in (v1, v2)
}


def get_prompt_set(name: str) -> PromptSet:
    """Return the prompt set called ``name`` (``v1`` when empty)."""
    try:
        return PROMPT_SETS[name or "v1"]
    except KeyError:
        raise ValueError(f"Unknown prompt version {name!r}; use {' or '.join(PROMPT_SETS)}") from None
