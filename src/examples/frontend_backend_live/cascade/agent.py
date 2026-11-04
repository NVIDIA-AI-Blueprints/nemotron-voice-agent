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
from examples.frontend_backend_live.prompts import Prompts, PromptSet
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
    def from_config(
        cls, config: LiveConfig, frontend_endpoint: ModelEndpoint, prompts: Prompts, *, frontend_client=None
    ) -> LiveAgent:
        """Build the agent for ``config``: the provider named by ``provider``, the texts of ``prompts``.

        Args:
            config: The validated ``frontend.cascade`` block of ``config.yaml``.
            frontend_endpoint: The talker's model endpoint.
            prompts: The session's read of the example's prompt file: its prompt set and its holding line.
            frontend_client: Optional injected client for the frontend (tests, probes); None builds the provider's own.

        Returns:
            The wired agent.

        Raises:
            ValueError: No provider has the configured name; the message lists the known ones.
        """
        provider = pick(FRONTENDS, config.provider, "frontend provider")
        frontend = provider(frontend_endpoint, config.reliability, frontend_client)
        texts = prompts.cascade
        return cls(
            frontend,
            texts,
            Talker(frontend, texts, holding=prompts.lines.holding),
            CommentaryWriter(frontend, texts),
            TurnNormalizer(frontend, texts, config.normalizer) if config.normalizer.enabled else None,
        )

    async def close(self) -> None:
        """Release the frontend model's connections."""
        await self.frontend.close()


# The name the hosts (the cascade adapter, the UI bot) build the agent by.
build_agent = LiveAgent.from_config
