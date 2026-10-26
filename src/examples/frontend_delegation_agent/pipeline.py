# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend Delegation Agent: registry entry point (``examples.frontend_delegation_agent.pipeline:bot``).

The Realtime gateway authenticates the socket and routes
``?model=nvidia/nemotron-realtime-frontend-delegation`` here before it sends any event
(this pipeline owns the whole Realtime session); browser UI sessions come from the repository's
UI transport. The agent's logic lives in ``core/``; ``adapters/pipecat`` connects it to the client
and to the shared Pipecat speech services. The Hermes backend runs in its own container (the
FDA gateway, ``core/sidecar``), reached over ``FDA_BACKEND_URL``.

The registry finds ``prompts.yaml``, ``tools.yaml`` and ``services.*.yaml`` beside this module, so
it stays at the package root.
"""

from __future__ import annotations

from examples.frontend_delegation_agent.adapters.pipecat.bot import bot

__all__ = ["bot"]
