# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Agent: registry entry point (``examples.frontend_backend_agent.pipeline:bot``).

The Realtime gateway authenticates the socket and routes
``?model=nvidia/nemotron-realtime-frontend-backend`` here before it sends any
event (this pipeline owns the whole Realtime session). The agent's logic lives in
``core/``; ``adapters/pipecat`` connects it to the client and to the shared Pipecat speech
services (VAD, ASR, TTS) and serves ``bot``.

The registry finds ``prompts.yaml`` and ``services.*.yaml`` beside this module, so it
stays at the package root.
"""

from __future__ import annotations

from examples.frontend_backend_agent.adapters.pipecat.pipeline import bot

__all__ = ["bot"]
