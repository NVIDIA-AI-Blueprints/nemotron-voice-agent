# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator settings for the live endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LiveSettings:
    """Operator settings for the live endpoints, read from the environment."""

    api_token: str = ""
    max_sessions: int = 8
    max_session_seconds: int = 1800
    ice_urls: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> LiveSettings:
        """Read ``LIVE_API_TOKEN``, ``LIVE_MAX_SESSIONS``, ``LIVE_MAX_SESSION_SECONDS`` and ``LIVE_ICE_URLS``."""
        return cls(
            api_token=os.getenv("LIVE_API_TOKEN", ""),
            max_sessions=int(os.getenv("LIVE_MAX_SESSIONS", "8")),
            max_session_seconds=int(os.getenv("LIVE_MAX_SESSION_SECONDS", "1800")),
            ice_urls=tuple(u.strip() for u in os.getenv("LIVE_ICE_URLS", "").split(",") if u.strip()),
        )
