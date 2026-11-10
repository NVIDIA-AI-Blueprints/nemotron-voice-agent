# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ACP agent classes by the name ``backend.harness.agent`` gives, each loaded only when a core uses it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module

from .agent import AcpAgent

AgentLoader = Callable[[], type[AcpAgent]]

AGENTS: Mapping[str, AgentLoader] = {
    "acp": lambda: AcpAgent,
    "hermes": lambda: (
        import_module("examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes").HermesAgent
    ),
}
