# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's source (``backend.adapter: llm-agent``), built once per core.

Building it connects nothing, but it parses the block and resolves the conductor's model, so a bad block or a model
the catalog does not list fails the core's build. The conductor runs on a copy of that endpoint with its own
``temperature``, since every provider takes its sampling from the endpoint. The workers use an ``LlmSource`` built
from the ``worker`` block. It has no ``new_one_shot``, so it cannot serve stateless requests.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from delegation_core.backend.llm import LlmSource
from delegation_core.backend.llm.source import ProviderMaker
from delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from delegation_core.backend.providers import Backend
from delegation_core.config import ModelEndpoint

from .adapter import AgentAdapter
from .settings import AgentSettings


class AgentSource:
    """The backend LLM agent: a model conductor and LLM workers, with the catalog's endpoints and the tool sources."""

    name: ClassVar[str] = "llm-agent"

    def __init__(
        self,
        settings: AgentSettings,
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
        make_provider: ProviderMaker | None = None,
        make_conductor_provider: ProviderMaker | None = None,
    ):
        """Hold what each IM session's adapter is made of; nothing connects.

        Args:
            settings: The agent's settings.
            endpoints: A model name to its endpoint (``None``: the default); a ``LookupError`` for a model the catalog
                does not list.
            tools: A fresh set of the session's server tool sources, for each IM session (the workers run them).
            vocabulary: The conductor's instructions and control tools.
            make_provider: Makes a worker's provider on an endpoint; by default the worker settings' provider class.
            make_conductor_provider: Makes the conductor's provider on its endpoint; by default the conductor settings'
                provider class, with the worker settings' limits (a test injects scripted ones).

        Raises:
            ValueError: The catalog does not list the conductor's model.
        """
        self.settings, self.endpoints, self.tools, self.vocabulary = settings, endpoints, tools, vocabulary
        self.workers = LlmSource(settings.worker, endpoints=endpoints, tools=tools, make_provider=make_provider)
        self.conductor_endpoint = self._conductor_endpoint()
        provider, limits = settings.conductor.provider, settings.worker.limits
        self.make_conductor_provider = make_conductor_provider or (lambda endpoint: provider(endpoint, limits))

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> AgentSource:
        """Validate ``backend.llm-agent`` (``block``) and build the source.

        Raises:
            ValueError: The block has an unknown key, a value of the wrong kind or a name no table has, or the catalog
                does not list the conductor's model.
        """
        return cls(AgentSettings.parse(block), endpoints=endpoints, tools=tools, vocabulary=vocabulary)

    def _conductor_endpoint(self) -> ModelEndpoint:
        """The conductor's endpoint: its model's, with the conductor's own temperature when it sets one."""
        conductor = self.settings.conductor
        try:
            endpoint = self.endpoints(conductor.model)
        except LookupError:
            raise ValueError(
                f"backend.llm-agent.conductor.model {conductor.model!r} is not in the model catalog"
            ) from None
        if conductor.temperature is None:
            return endpoint
        return replace(endpoint, extra_params={**endpoint.extra_params, "temperature": conductor.temperature})

    def describe(self) -> dict:
        """What it serves, as plain JSON for ``GET /v1/live/info``: no credentials."""
        default, worker = self.endpoints(None), self.settings.worker
        return {
            "adapter": self.name,
            "model": default.model,
            "base_url": default.base_url,
            "max_running": self.settings.max_running,
            "conductor": {**self.settings.conductor.describe(), "model": self.conductor_endpoint.model},
            "worker": {
                "provider": worker.provider.name,
                "unlisted_models": worker.unlisted_models.name,
                "max_tool_rounds": worker.max_tool_rounds,
            },
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if the workers cannot serve ``model``, the model a request names."""
        await self.workers.check_model(model)

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings the workers' provider cannot honor, as ``(param, message)`` pairs (as ``llm``'s)."""
        return self.workers.unsupported(settings)

    def conductor_provider(self) -> Backend:
        """A new provider for one session's conductor, on the conductor's endpoint."""
        return self.make_conductor_provider(self.conductor_endpoint)

    def new(self) -> AgentAdapter:
        """The backend of a new IM session."""
        return AgentAdapter(self)

    async def aclose(self) -> None:
        """Release what the workers' source holds; each adapter closes its own providers."""
        await self.workers.aclose()
