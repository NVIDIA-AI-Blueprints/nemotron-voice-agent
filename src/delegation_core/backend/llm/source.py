# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM's source (``backend.adapter: llm``), built once per core; also the default one-shot source.

Building it connects nothing; model clients are made at a session's first round, one provider per model per session.
A session's adapter gets a fresh set of the server's tool sources and two attempts per round; a one-shot adapter gets
one attempt per round and no server tools.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from delegation_core.backend.providers import Backend
from delegation_core.config import ModelEndpoint

from .adapter import LlmAdapter
from .settings import LlmSettings

# Tries per round: one retry of an attempt that dispatched nothing and delivered nothing.
ATTEMPTS = 2
# A stateless request's round is tried once.
ONE_SHOT_ATTEMPTS = 1

ProviderMaker = Callable[[ModelEndpoint], Backend]


class LlmSource:
    """The backend LLM, directly connected: its settings, the catalog's endpoints and the session's tool sources."""

    name: ClassVar[str] = "llm"

    def __init__(
        self,
        settings: LlmSettings,
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        make_provider: ProviderMaker | None = None,
    ):
        """Hold what each IM session's adapter is made of; nothing connects.

        Args:
            settings: The backend LLM's settings.
            endpoints: A model name to its endpoint (``None``: the default); a ``LookupError`` for a model the catalog
                does not list.
            tools: A fresh set of the session's server tool sources, for each IM session.
            make_provider: Makes a model provider on an endpoint; by default the settings' provider class with the
                settings' limits.
        """
        self.settings, self.endpoints, self.tools = settings, endpoints, tools
        self.make_provider = make_provider or (lambda endpoint: settings.provider(endpoint, settings.limits))

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> LlmSource:
        """Validate ``backend.llm`` (``block``) and build the source; a rule conductor reads no ``vocabulary``.

        Raises:
            ValueError: The block has an unknown key, a value of the wrong kind, or a name no table has.
        """
        return cls(LlmSettings.parse(block), endpoints=endpoints, tools=tools)

    def describe(self) -> dict:
        """What it serves, as plain JSON for ``GET /v1/live/info``: no credentials."""
        default = self.endpoints(None)
        return {
            "adapter": self.name,
            **self.settings.describe(),
            "model": default.model,
            "base_url": default.base_url,
            "max_running": LlmAdapter.max_running,
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if it cannot serve ``model``: the catalog does not list it and ``report`` holds."""
        self.endpoint(model)

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings its provider cannot honor, as ``(param, message)`` pairs."""
        return self.settings.provider.unsupported(settings)

    def endpoint(self, model: str | None) -> ModelEndpoint:
        """The endpoint ``model`` runs on, by the catalog and the unlisted-model policy.

        Raises:
            ValueError: The catalog does not list ``model`` and the policy reports it.
        """
        return self.settings.unlisted_models.endpoint(model, self.endpoints)

    def new(self) -> LlmAdapter:
        """The backend of a new IM session."""
        return LlmAdapter(self, tools=self.tools(), attempts=ATTEMPTS)

    def new_one_shot(self) -> LlmAdapter:
        """The backend of one stateless request's one-shot session."""
        return LlmAdapter(self, tools=(), attempts=ONE_SHOT_ATTEMPTS)

    async def aclose(self) -> None:
        """Nothing to release: each adapter closes its own providers."""
