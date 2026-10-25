# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy ``FDH_*`` environment variables: the check both FDA config loaders run (stdlib only).

The example was Frontend Delegation Hermes (``FDH_*`` variables) and is now Frontend Delegation
Agent (``FDA_*``). A stale ``FDH_*`` variable would otherwise be ignored without a word. Called only
from ``load_delegation_config`` and ``load_gateway_config``, never at import time, so a stale ``.env``
that every example's container reads cannot affect the other examples.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

OLD_PREFIX = "FDH_"


def legacy_env_error(environ: Mapping[str, str] | None = None) -> str | None:
    """The error text naming every set ``FDH_*`` variable (names only, never values), or ``None``."""
    found = sorted(name for name in (os.environ if environ is None else environ) if name.startswith(OLD_PREFIX))
    if not found:
        return None
    return (
        f"FDH_* variables were renamed to FDA_* (found: {', '.join(found)}); "
        "rename them in .env, the shell or the Compose overrides"
    )
