# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Read/write classification of a caller-declared tool from its own name and description.

Write safety depends on this split: argument repair only ever touches a *read*
tool, so the classification must be conservative. A tool is a read only when its
name and the opening verb of its description both say so and no later word of
its name names a change; everything else, including any tool the rules cannot
place, is a write.
"""

from __future__ import annotations

import re
from typing import Literal

READ_NAME_PREFIXES = ("get_", "find_", "search_", "list_", "lookup_", "check_")
MUTATING_NAME_TOKENS = frozenset(
    {
        "book",
        "cancel",
        "create",
        "update",
        "modify",
        "delete",
        "remove",
        "add",
        "send",
        "pay",
        "charge",
        "transfer",
        "reset",
        "enable",
        "disable",
        "set",
        "submit",
        "in",
    }
)

_READ_DESCRIPTION_OPENING = re.compile(r"(?:get|find|search|list|look\s+up|lookup|check)\b", re.IGNORECASE)


def tool_kind(name: str, description: str) -> Literal["read", "write"]:
    """``"read"`` for a lookup by name prefix, description verb and name tokens; otherwise ``"write"``.

    ``check_in_member`` is a write ("in" names a change), and so is a ``lookup_*``
    tool whose description opens "Updates". Only the description's opening verb
    is read, so "Get the items in an order" stays a read.
    """
    if not isinstance(name, str) or not isinstance(description, str):
        return "write"
    lowered = name.lower()
    if not lowered.startswith(READ_NAME_PREFIXES):
        return "write"
    if _READ_DESCRIPTION_OPENING.match(description.lstrip()) is None:
        return "write"
    if any(token in MUTATING_NAME_TOKENS for token in lowered.split("_")[1:]):
        return "write"
    return "read"
