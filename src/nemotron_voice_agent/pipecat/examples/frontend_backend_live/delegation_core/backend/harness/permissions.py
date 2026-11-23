# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The answer to an agent's permission requests (``backend.harness.permission``): never an approval."""

from __future__ import annotations

from collections.abc import Sequence

# The option kinds that decline, the narrowest first (ACP's PermissionOptionKind).
DECLINING = ("reject_once", "reject_always")


class Reject:
    """``reject``: decline every request, or answer it cancelled when no option declines."""

    def choose(self, options: Sequence[tuple[str, str]]) -> str | None:
        """Return the id of the narrowest declining option, else ``None`` (cancelled)."""
        by_kind = {kind: option_id for option_id, kind in reversed(options)}
        return next((by_kind[kind] for kind in DECLINING if kind in by_kind), None)
