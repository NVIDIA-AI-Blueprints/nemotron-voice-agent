# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``hermes:`` section of the gateway config (read only when ``backend.kind: hermes``). Stdlib only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class HermesSettings:
    """Settings only Hermes uses; the shared ones (model, timeouts, prompts) are in ``backend:``."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "provider": "custom",
        "context_length": 131072,
        "load_soul_identity": True,
        "disable_streaming": False,
    }

    provider: str
    #: Written to each worker's ``HERMES_HOME/config.yaml`` (Hermes' 64K floor).
    context_length: int
    #: Required with ``skip_context_files`` or ``SOUL.md`` is not loaded.
    load_soul_identity: bool
    #: Non-streaming test servers only (Hermes streams by default).
    disable_streaming: bool

    @classmethod
    def parse(cls, pick: Callable[[str, type], Any]) -> HermesSettings:
        """Typed fields; ``pick(key, type)`` reads and converts one key of the section."""
        return cls(
            provider=pick("provider", str),
            context_length=pick("context_length", int),
            load_soul_identity=pick("load_soul_identity", bool),
            disable_streaming=pick("disable_streaming", bool),
        )

    def worker_settings(self) -> dict[str, Any]:
        """The fields the Hermes worker reads from ``configure``."""
        return {
            "provider": self.provider,
            "context_length": self.context_length,
            "load_soul_identity": self.load_soul_identity,
            "disable_streaming": self.disable_streaming,
        }

    def summary(self) -> dict[str, Any]:
        """For logs and ``/health``."""
        return {"provider": self.provider, "context_length": self.context_length}
