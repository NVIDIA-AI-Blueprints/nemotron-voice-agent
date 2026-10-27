# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How the example turns a name in ``config.yaml`` into an implementation: one plain mapping per table.

Each extension point of the frontend adapter has one table, a plain ``dict`` from the name the settings use to the
class (``FRONTEND_ADAPTERS``, ``FRONTENDS``, the prompt sets, the result wordings). Only the composition root and a
class's own ``Settings.parse`` or ``from_config`` pick from a table, so no code compares a kind against a name. The
delegation core has its own copy (``delegation_core/tables.py``): the two boxes share no module.
"""

from __future__ import annotations

from collections.abc import Mapping


def pick[T](table: Mapping[str, T], name: str, kind: str) -> T:
    """The entry ``name`` of ``table``.

    Raises:
        ValueError: ``Unknown <kind> 'x'; known: a, b``, so a typo in the settings names what is available.
    """
    try:
        return table[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown {kind} {name!r}; known: {', '.join(sorted(table)) or 'none'}") from None
