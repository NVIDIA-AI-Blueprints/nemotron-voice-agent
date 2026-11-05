# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name tables: each extension point maps the names its settings use to the implementations.

Only a builder and a class's own ``from_settings`` (or ``parse``, ``from_config``) pick from a table, so each choice is
made once, when it is built. The example's frontend picks from its own tables with :func:`pick` too.
"""

from __future__ import annotations

from collections.abc import Mapping


def pick[T](table: Mapping[str, T], name: str, kind: str) -> T:
    """Return the entry ``name`` of ``table``.

    Raises:
        ValueError: ``Unknown <kind> 'x'; known: a, b``, so a typo in the settings names what is available.
    """
    try:
        return table[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown {kind} {name!r}; known: {', '.join(sorted(table)) or 'none'}") from None
