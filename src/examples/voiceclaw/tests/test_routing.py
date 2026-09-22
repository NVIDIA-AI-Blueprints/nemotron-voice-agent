# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import pytest

from voiceclaw.application.routing import build_turn_routing_policy
from voiceclaw.ports.runtime import TurnDirectiveKind


def test_unknown_routing_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported turn routing mode"):
        build_turn_routing_policy("unknown")


def test_model_selected_policy_keeps_automatic_tool_selection() -> None:
    directive = build_turn_routing_policy("model").decide("Delegate this if useful")

    assert directive.kind is TurnDirectiveKind.AUTO
    assert directive.logical_tool is None
