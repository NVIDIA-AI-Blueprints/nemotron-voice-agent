# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the talker, commentary writer and thinker from ``config.yaml`` and resolved model endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from examples.frontend_backend_live.cascade.commentary.writer import CommentaryWriter
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.delegation.thinker import Thinker
from examples.frontend_backend_live.models import BACKENDS, FRONTENDS, Backend, Frontend
from examples.frontend_backend_live.prompts import PromptSet, get_prompt_set


@dataclass
class LiveAgent:
    """The model-facing parts of one call."""

    frontend: Frontend
    backend: Backend
    prompts: PromptSet
    talker: Talker
    writer: CommentaryWriter
    thinker: Thinker

    async def close(self) -> None:
        """Release both models' connections."""
        await self.frontend.close()
        await self.backend.close()


def build_agent(
    config: LiveConfig,
    frontend_endpoint: ModelEndpoint,
    backend_endpoint: ModelEndpoint,
    *,
    frontend_client=None,
    backend_client=None,
) -> LiveAgent:
    """Build the agent for ``config``: each role's implementation is looked up by its configured name.

    Args:
        config: The validated ``config.yaml``.
        frontend_endpoint: The talker's model endpoint.
        backend_endpoint: The thinker's model endpoint.
        frontend_client: Optional injected client for the frontend (tests, probes).
        backend_client: Optional injected client for the backend.

    Returns:
        The wired agent.

    Raises:
        ValueError: A configured provider is not registered.
    """
    frontend_cls, backend_cls = FRONTENDS.get(config.frontend.provider), BACKENDS.get(config.backend.provider)
    frontend = (
        frontend_cls(frontend_endpoint, config.reliability, frontend_client)
        if frontend_client is not None
        else frontend_cls(frontend_endpoint, config.reliability)
    )
    backend = (
        backend_cls(backend_endpoint, config.reliability, backend_client)
        if backend_client is not None
        else backend_cls(backend_endpoint, config.reliability)
    )
    prompts = get_prompt_set(config.prompt_version)
    return LiveAgent(
        frontend,
        backend,
        prompts,
        Talker(frontend, prompts),
        CommentaryWriter(frontend, prompts),
        Thinker(backend, prompts),
    )
