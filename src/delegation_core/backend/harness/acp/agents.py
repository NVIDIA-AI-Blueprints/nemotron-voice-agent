# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ACP agent classes by the name ``backend.harness.agent`` gives; each loads only when a core is built with it.

A new agent class is a package beside ``hermes`` plus one entry here; a plain ACP agent needs neither.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module

from .agent import AcpAgent

# Loads one agent class, on first use.
AgentLoader = Callable[[], type[AcpAgent]]

AGENTS: Mapping[str, AgentLoader] = {
    "acp": lambda: AcpAgent,
    "hermes": lambda: import_module("delegation_core.backend.harness.acp.hermes").HermesAgent,
}
