# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend registry: ``backend.kind`` -> :class:`BackendSpec` (llm-backend-plan.md section 2).

To add a backend: create ``core/backends/<kind>/`` with a ``SPEC``, add it below, add a
``<kind>:`` section to ``config/gateway.yaml`` if it has settings, and add
``config/prompts/<profile>/backend/<kind>.yaml`` for every prompt profile.

Importing this module imports only the specs (stdlib); hosts load when a worker builds one.
"""

from __future__ import annotations

from collections.abc import Mapping

from examples.frontend_delegation_hermes.core.backends import fake, hermes, llm
from examples.frontend_delegation_hermes.core.backends.base import BackendSpec, Placement, load

BACKENDS: Mapping[str, BackendSpec] = {spec.kind: spec for spec in (hermes.SPEC, llm.SPEC, fake.SPEC)}

#: The backend kinds, in registry order.
KINDS: tuple[str, ...] = tuple(BACKENDS)


class UnknownBackendError(ValueError):
    """``backend.kind`` names no registered backend."""


def get_spec(kind: str) -> BackendSpec:
    """The spec of ``kind``; raises :class:`UnknownBackendError` with the valid kinds."""
    spec = BACKENDS.get(kind)
    if spec is None:
        raise UnknownBackendError(f"unknown backend kind {kind!r}; expected one of {KINDS}")
    return spec


__all__ = ["BACKENDS", "KINDS", "BackendSpec", "Placement", "UnknownBackendError", "get_spec", "load"]
