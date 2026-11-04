# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lines the voice says word for word: ``agent_prompts.Lines`` of the example's ``prompts.yaml``.

An empty line says nothing. Each host hands the cascade the lines of its session's read of the file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class FixedLines:
    """Every line the voice says word for word."""

    apology: str  # a delegation failed and no tool ran
    unclear: str  # a delegation failed, or came back late, after a tool ran: the outcome is not known
    lost: str  # the delegation's stream ended with no ending
    refusal: str  # the delegation was refused before any work
    app_wait_line: str  # the app has not answered a delegation's calls yet
    holding: str  # the talker failed while work is open
    still_working: str  # the Live Console's progress line for a slow function batch; the cascade says none
    intro: str  # the greeting instructions: the UI bot's, and the Live Console's "Greet first" preset
    greeting: str  # the UI bot's greeting when its instructions cannot be worded
    app_started: str  # the Live Console's silent note that a client delegation's work started; never said

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> FixedLines:
        """Validate ``block``: every name present, no other, each a string.

        Raises:
            ValueError: A name is missing or unknown, or a line is not a string.
        """
        names = {field.name for field in fields(cls)}
        if not isinstance(block, Mapping):
            raise ValueError("agent_prompts.Lines must be a mapping of name to text")
        unknown, missing = set(block) - names, names - set(block)
        if unknown or missing:
            raise ValueError(f"agent_prompts.Lines: unknown {sorted(unknown)}, missing {sorted(missing)}")
        wrong = sorted(name for name, text in block.items() if not isinstance(text, str))
        if wrong:
            raise ValueError(f"agent_prompts.Lines: {', '.join(wrong)} must be text")
        return cls(**block)
