# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The emulator backend: a scripted ``EmulatorAgent`` that emulates Hermes for tests (no model, no credentials)."""

from __future__ import annotations

from examples.frontend_delegation_agent.core.backends.base import BackendSpec, Placement

SPEC = BackendSpec(
    kind="emulator",
    host="examples.frontend_delegation_agent.core.backends.emulator.host:EmulatorAgentHost",
    placements=frozenset({Placement.WORKER_PROCESS, Placement.IN_PROCESS}),
    prompt_kind="hermes",
    worker_setup="examples.frontend_delegation_agent.core.backends.emulator.setup:EmulatorWorkerSetup",
)
