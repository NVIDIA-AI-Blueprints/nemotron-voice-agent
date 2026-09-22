# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Provider-neutral policy boundary for one finalized user turn."""

from __future__ import annotations

from typing import Protocol

from voiceclaw.ports.runtime import TurnDirective


class TurnRoutingPolicy(Protocol):
    """Choose a provider-neutral action for a finalized user turn."""

    def decide(self, text: str) -> TurnDirective:
        """Return one deterministic routing directive."""
        ...


class ModelSelectedTurnRoutingPolicy:
    """Leave ordinary turn selection to the configured frontend model."""

    def decide(self, text: str) -> TurnDirective:
        """Return an automatic directive for backwards-compatible model routing."""
        del text
        return TurnDirective.auto(reason_code="frontend_model")


def build_turn_routing_policy(mode: str) -> TurnRoutingPolicy:
    """Build the configured routing policy without binding it to a provider."""
    if mode == "model":
        return ModelSelectedTurnRoutingPolicy()
    raise ValueError(f"unsupported turn routing mode: {mode}")


__all__ = [
    "ModelSelectedTurnRoutingPolicy",
    "TurnRoutingPolicy",
    "build_turn_routing_policy",
]
