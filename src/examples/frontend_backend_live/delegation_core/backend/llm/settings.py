# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM's settings: the ``backend.llm`` block and the defaults of every key.

:class:`LlmSettings` checks the block when the core is built, so an unknown key, a fraction where a whole number is due,
or an unknown name fails the build.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, ConfigDict

from examples.frontend_backend_live.delegation_core.backend.providers import (
    BACKENDS,
    Backend,
    ChatCompletionsBackend,
    ProviderLimits,
)
from examples.frontend_backend_live.delegation_core.config import Count, Seconds, Settings, WholeNumber
from examples.frontend_backend_live.delegation_core.tables import pick


def provider_class(name: Any) -> Any:
    """The provider class a name of ``providers.BACKENDS`` gives; a class as it is."""
    return pick(BACKENDS, name, "backend provider") if isinstance(name, str) else name


# A model provider, named in the settings: chat-completions | openai-responses (providers.BACKENDS).
Provider = Annotated[type[Backend], BeforeValidator(provider_class)]


class LlmSettings(Settings):
    """The backend LLM's settings, with its provider's name turned into the provider class."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm")

    provider: Provider = ChatCompletionsBackend  # chat-completions | openai-responses
    max_tool_rounds: WholeNumber = 30  # rounds before the capped round, whose instructions add texts.round_cap_note
    provider_retries: Count = 2  # tries after the first, when opening a round fails with a transient error
    first_event_timeout_seconds: Seconds = 45.0  # from a round's request to its first event
    idle_timeout_seconds: Seconds = 30.0  # the longest silence between two events of a round
    server_tool_timeout_seconds: Seconds = 60.0  # one call a tool source runs
    stop_timeout_seconds: Seconds = 10.0  # how long a stop waits for the round and the started calls to end

    @property
    def limits(self) -> ProviderLimits:
        """What a provider of these settings opens and times its rounds by."""
        return ProviderLimits(self.provider_retries, self.first_event_timeout_seconds, self.idle_timeout_seconds)

    def describe(self) -> dict:
        """The settings as plain JSON, by name."""
        return {"provider": self.provider.name, "max_tool_rounds": self.max_tool_rounds}
