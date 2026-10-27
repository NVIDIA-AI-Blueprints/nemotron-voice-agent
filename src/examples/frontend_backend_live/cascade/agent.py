# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the talker, the writer and the normalizer from ``config.yaml`` and the frontend's model endpoint.

The backend model belongs to the call's delegation core, not to this agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.frontend_backend_live.cascade.normalizer import TurnNormalizer
from examples.frontend_backend_live.cascade.talker.talker import Talker
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from examples.frontend_backend_live.models import FRONTENDS, Frontend
from examples.frontend_backend_live.prompts import PromptSet, get_prompt_set
from examples.frontend_backend_live.prompts.catalog import catalog_line
from examples.frontend_backend_live.tables import pick


@dataclass
class LiveAgent:
    """The frontend's model-facing parts of one call."""

    frontend: Frontend
    prompts: PromptSet
    talker: Talker
    writer: CommentaryWriter
    normalizer: TurnNormalizer | None = None

    @classmethod
    def from_config(cls, config: LiveConfig, frontend_endpoint: ModelEndpoint, *, frontend_client=None) -> LiveAgent:
        """Build the agent for ``config``: the provider named by ``provider``, the set by ``prompt_version``.

        Args:
            config: The validated ``frontend.cascade`` block of ``config.yaml``.
            frontend_endpoint: The talker's model endpoint.
            frontend_client: Optional injected client for the frontend (tests, probes); None builds the provider's own.

        Returns:
            The wired agent.

        Raises:
            ValueError: No provider or no prompt set has the configured name; the message lists the known ones.
        """
        provider = pick(FRONTENDS, config.provider, "frontend provider")
        frontend = provider(frontend_endpoint, config.reliability, frontend_client)
        prompts = get_prompt_set(config.prompt_version)
        return cls(
            frontend,
            prompts,
            Talker(frontend, prompts, holding=catalog_line("holding")),
            CommentaryWriter(frontend, prompts),
            TurnNormalizer(frontend, prompts, config.normalizer) if config.normalizer.enabled else None,
        )

    async def close(self) -> None:
        """Release the frontend model's connections."""
        await self.frontend.close()


# The name the hosts (the live adapter, the UI bot) and the tests build the agent by.
build_agent = LiveAgent.from_config
