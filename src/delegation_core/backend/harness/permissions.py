# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The answer to an agent's permission requests (``backend.harness.permission``); no approval reaches the caller.

An agent asks only before one of its own tools that needs approval, such as a terminal command or a file edit; the
session's bridged tools never ask.
"""

from __future__ import annotations

from collections.abc import Sequence

# The option kinds that decline, the narrowest first (ACP's PermissionOptionKind).
DECLINING = ("reject_once", "reject_always")


class Reject:
    """``reject``: decline every request; with no declining option, answer it cancelled."""

    def choose(self, options: Sequence[tuple[str, str]]) -> str | None:
        """The first declining option's id, the narrowest kind first, else ``None`` (cancelled)."""
        by_kind = {kind: option_id for option_id, kind in reversed(options)}
        return next((by_kind[kind] for kind in DECLINING if kind in by_kind), None)
