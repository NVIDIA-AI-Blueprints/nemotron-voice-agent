# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lines the voice says word for word: the ``lines:`` mapping of the example's ``prompts.yaml``.

An empty line says nothing. A host may hand the cascade its own set; otherwise the catalog's is read once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cache
from typing import Any

from examples.frontend_backend_live.prompts.catalog import CATALOG_PATH
from utils import load_yaml_file


@dataclass(frozen=True)
class FixedLines:
    """Every line the voice says word for word."""

    apology: str  # a delegation failed and no tool ran
    unclear: str  # a delegation failed, or came back late, after a tool ran: the outcome is not known
    lost: str  # the delegation's stream ended with no ending
    refusal: str  # the delegation was refused before any work
    app_wait_line: str  # the app has not answered a delegation's calls yet
    lead_in: str  # the acknowledgement of a delegation whose talker decision said none
    holding: str  # the talker failed while work is open
    still_working: str  # the quiet-spell line when a delegation has no progress line to say
    intro: str  # the UI bot's greeting instructions

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> FixedLines:
        """Validate ``block``: every name present, no other, each a string.

        Raises:
            ValueError: A name is missing or unknown, or a line is not a string.
        """
        names = {field.name for field in fields(cls)}
        if not isinstance(block, Mapping):
            raise ValueError("lines must be a mapping of name to text")
        unknown, missing = set(block) - names, names - set(block)
        if unknown or missing:
            raise ValueError(f"lines: unknown {sorted(unknown)}, missing {sorted(missing)}")
        wrong = sorted(name for name, text in block.items() if not isinstance(text, str))
        if wrong:
            raise ValueError(f"lines: {', '.join(wrong)} must be text")
        return cls(**block)


@cache
def catalog_lines() -> FixedLines:
    """The example's lines, read once from its ``prompts.yaml``."""
    return FixedLines.parse(load_yaml_file(CATALOG_PATH, required=True).get("lines", {}))
