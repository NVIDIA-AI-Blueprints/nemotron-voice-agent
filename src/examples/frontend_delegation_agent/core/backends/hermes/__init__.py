# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Hermes backend: one Hermes ``AIAgent`` per worker process (Python 3.14 image, OpenShell by default).

Only ``host.py`` imports Hermes; this module and ``settings.py`` are stdlib only.
"""

from __future__ import annotations

from examples.frontend_delegation_agent.core.backends.base import BackendSpec, Placement

_PACKAGE = "examples.frontend_delegation_agent.core.backends.hermes"

SPEC = BackendSpec(
    kind="hermes",
    host=f"{_PACKAGE}.host:HermesAgentHost",
    placements=frozenset({Placement.WORKER_PROCESS}),
    settings=f"{_PACKAGE}.settings:HermesSettings",
    worker_setup=f"{_PACKAGE}.home:HermesWorkerSetup",
)
