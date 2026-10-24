# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fake backend: a scripted ``FakeAgent`` for tests, CI and ``--stub-backend`` (no model, no credentials)."""

from __future__ import annotations

from examples.frontend_delegation_hermes.core.backends.base import BackendSpec, Placement

SPEC = BackendSpec(
    kind="fake",
    host="examples.frontend_delegation_hermes.core.backends.fake.host:FakeAgentHost",
    placements=frozenset({Placement.WORKER_PROCESS, Placement.IN_PROCESS}),
    prompt_kind="hermes",
    worker_setup="examples.frontend_delegation_hermes.core.backends.fake.setup:FakeWorkerSetup",
)
