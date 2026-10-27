# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM's settings: the ``backend.llm`` block and the defaults of every key.

:meth:`LlmSettings.parse` checks the block when the core is built, so an unknown key, a fraction where a whole number
is due, or an unknown name fails the build. ``unlisted_models`` decides what happens to a requested model the catalog
does not list: ``report`` refuses it with a nonfatal ``responses_validation_failed``, and ``fallback`` runs it on the
configured endpoint under its own name and logs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from loguru import logger

from delegation_core.backend.conductors import RULE_CONDUCTORS, StartEach, SteerOpenWork
from delegation_core.backend.port import EndpointLookup
from delegation_core.backend.providers import BACKENDS, Backend, ChatCompletionsBackend, ProviderLimits
from delegation_core.config import ModelEndpoint
from delegation_core.tables import pick


class ReportUnlisted:
    """``report``: a model the catalog does not list is refused."""

    name: ClassVar[str] = "report"

    def endpoint(self, model: str | None, endpoints: EndpointLookup) -> ModelEndpoint:
        """The model's endpoint.

        Raises:
            ValueError: The catalog does not list ``model``.
        """
        try:
            return endpoints(model)
        except LookupError:
            raise ValueError(
                f"The backend does not serve the model {model!r}: it is not in the model catalog"
            ) from None


class FallBackToDefault:
    """``fallback``: a model the catalog does not list runs on the configured endpoint, under its own name."""

    name: ClassVar[str] = "fallback"

    def endpoint(self, model: str | None, endpoints: EndpointLookup) -> ModelEndpoint:
        """The model's endpoint, or the configured one for a model the catalog does not list (logged)."""
        try:
            return endpoints(model)
        except LookupError:
            default = endpoints(None)
            logger.warning(
                f"backend.unlisted_model model={model!r}: runs on the configured endpoint {default.base_url}"
            )
            return replace(default, model=model)


UnlistedModels = ReportUnlisted | FallBackToDefault
# Each policy for an unlisted model by its name in backend.llm.unlisted_models.
UNLISTED_MODELS: Mapping[str, UnlistedModels] = {
    policy.name: policy for policy in (ReportUnlisted(), FallBackToDefault())
}

# The keys of whole numbers, with their least value; the keys of durations in seconds; the keys of text.
COUNTS = {"max_tool_rounds": 1, "provider_retries": 0}
SECONDS = (
    "first_event_timeout_seconds",
    "idle_timeout_seconds",
    "server_tool_timeout_seconds",
    "stop_timeout_seconds",
)
TEXTS = ("round_cap_note",)


@dataclass(frozen=True)
class LlmSettings:
    """The backend LLM's settings, with every name already turned into what it names."""

    provider: type[Backend] = ChatCompletionsBackend  # chat-completions | openai-responses (providers.BACKENDS)
    conductor: type[SteerOpenWork] | type[StartEach] = SteerOpenWork  # steer | start_each (RULE_CONDUCTORS)
    unlisted_models: UnlistedModels = UNLISTED_MODELS["report"]  # report | fallback
    max_tool_rounds: int = 8  # rounds before the capped round
    provider_retries: int = 2  # tries after the first, when opening a round fails with a transient error
    first_event_timeout_seconds: float = 45.0  # from a round's request to its first event
    idle_timeout_seconds: float = 30.0  # the longest silence between two events of a round
    server_tool_timeout_seconds: float = 60.0  # one call a tool source runs
    stop_timeout_seconds: float = 10.0  # how long a stop waits for the round and the started calls to end
    round_cap_note: str = "You have reached the step limit. Answer the caller with what you have."

    @property
    def limits(self) -> ProviderLimits:
        """What a provider of these settings opens and times its rounds by."""
        return ProviderLimits(self.provider_retries, self.first_event_timeout_seconds, self.idle_timeout_seconds)

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> LlmSettings:
        """Validate the ``backend.llm`` block and turn its names into their classes.

        Raises:
            ValueError: A key the backend LLM does not have, a value of the wrong kind, or a name no table has.
        """
        if not isinstance(block, Mapping):
            raise ValueError("backend.llm must be a mapping")
        unknown = set(block) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"backend.llm has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        values: dict[str, Any] = {}
        if "provider" in block:
            values["provider"] = pick(BACKENDS, block["provider"], "backend provider")
        if "conductor" in block:
            values["conductor"] = pick(RULE_CONDUCTORS, block["conductor"], "llm conductor")
        if "unlisted_models" in block:
            values["unlisted_models"] = pick(UNLISTED_MODELS, block["unlisted_models"], "unlisted-model policy")
        for key, least in COUNTS.items():
            if key in block:
                value = block[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < least:
                    raise ValueError(f"backend.llm.{key} must be a whole number of at least {least}")
                values[key] = value
        for key in SECONDS:
            if key in block:
                value = block[key]
                if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                    raise ValueError(f"backend.llm.{key} must be a positive number of seconds")
                values[key] = float(value)
        for key in TEXTS:
            if key in block:
                if not isinstance(block[key], str) or not block[key].strip():
                    raise ValueError(f"backend.llm.{key} must be non-empty text")
                values[key] = block[key]
        return cls(**values)

    def describe(self) -> dict:
        """The settings as plain JSON, by name."""
        return {
            "provider": self.provider.name,
            "conductor": self.conductor.name,
            "unlisted_models": self.unlisted_models.name,
            "max_tool_rounds": self.max_tool_rounds,
        }
