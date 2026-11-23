# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name tables: each extension point maps the names its settings use to the implementations."""

from __future__ import annotations

from collections.abc import Mapping


def pick[T](table: Mapping[str, T], name: str, kind: str) -> T:
    """Return the entry ``name`` of ``table``; ``ValueError`` names the known entries, so a typo shows the choices."""
    try:
        return table[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown {kind} {name!r}; known: {', '.join(sorted(table)) or 'none'}") from None
