# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM's settings: the ``backend.llm`` block and the defaults of every key."""

from __future__ import annotations

from pydantic import ConfigDict

from examples.frontend_backend_live.delegation_core.backend.providers import ProviderLimits
from examples.frontend_backend_live.delegation_core.config import Count, Seconds, Settings, WholeNumber


class LlmSettings(Settings):
    """The backend LLM's settings; each model's provider is picked by the API its catalog entry names."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="backend.llm")

    # Rounds before the capped round, whose instructions add texts.round_cap_note.
    max_tool_rounds: WholeNumber = 30
    # Tries after the first, for a round that failed in a way that can clear up: one count for every such failure,
    # restarts for steering aside. A stateless request's round is tried again only after a transient error before
    # its first event.
    provider_retries: Count = 2
    # From a round's request to its first event.
    first_event_timeout_seconds: Seconds = 45.0
    # The longest silence between two events of a round.
    idle_timeout_seconds: Seconds = 30.0
    # One call a tool source runs.
    server_tool_timeout_seconds: Seconds = 60.0
    # How long a stop waits for the round and the started calls to end.
    stop_timeout_seconds: Seconds = 10.0

    @property
    def limits(self) -> ProviderLimits:
        """The limits a provider of these settings opens and times its rounds by."""
        return ProviderLimits(self.provider_retries, self.first_event_timeout_seconds, self.idle_timeout_seconds)

    def describe(self) -> dict:
        """Return the settings as plain JSON, by name."""
        return {"max_tool_rounds": self.max_tool_rounds}
