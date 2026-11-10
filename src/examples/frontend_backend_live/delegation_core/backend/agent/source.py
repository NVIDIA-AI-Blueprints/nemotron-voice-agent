# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's source (``backend.adapter: llm-agent``), built once per core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from examples.frontend_backend_live.delegation_core.backend.llm import LlmSource
from examples.frontend_backend_live.delegation_core.backend.llm.source import ProviderMaker
from examples.frontend_backend_live.delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from examples.frontend_backend_live.delegation_core.backend.providers import Backend, provider_class, provider_for
from examples.shared.delegation_contract import ModelEndpoint

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
        """Hold what each IM session's adapter is made of and resolve the conductor's model; nothing connects."""
        self.settings, self.endpoints, self.tools, self.vocabulary = settings, endpoints, tools, vocabulary
        self.workers = LlmSource(settings.worker, endpoints=endpoints, tools=tools, make_provider=make_provider)
        self.conductor_endpoint = self._conductor_endpoint()
        # Checked here, so an unknown API on the conductor model's entry fails the core's build.
        provider_class(self.conductor_endpoint)
        limits = settings.worker.limits
        self.make_conductor_provider = make_conductor_provider or (lambda endpoint: provider_for(endpoint, limits))

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> AgentSource:
        """Validate the ``backend.llm-agent`` block and build the source.

        Raises:
            ValueError: The block is invalid, the catalog does not list the conductor's model, or the entry of the
                default or the conductor's model names an API that no provider speaks.
        """
        return cls(AgentSettings.model_validate(block), endpoints=endpoints, tools=tools, vocabulary=vocabulary)

    def _conductor_endpoint(self) -> ModelEndpoint:
        """Return the conductor model's endpoint, with the conductor's temperature, which providers read from it."""
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
        """Describe what it serves as plain JSON, without credentials.

        ``conductor`` holds the model conductor's settings, or the name of the rule that places requests instead.
        """
        default, worker = self.endpoints(None), self.settings.worker
        rule = AgentAdapter.rule(self.settings.max_running)
        endpoint = self.conductor_endpoint
        conductor = {**self.settings.conductor.describe(), "model": endpoint.model, "api": endpoint.api}
        return {
            "adapter": self.name,
            "model": default.model,
            "base_url": default.base_url,
            "api": default.api,
            "max_running": self.settings.max_running,
            "conductor": rule.name if rule is not None else conductor,
            "worker": worker.describe(),
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if the workers cannot serve ``model``, the model a request names."""
        await self.workers.check_model(model)

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings the provider of their model cannot honor, as ``(param, message)`` pairs."""
        return self.workers.unsupported(settings)

    def conductor_provider(self) -> Backend:
        """Make a provider for one session's conductor, on the conductor's endpoint."""
        return self.make_conductor_provider(self.conductor_endpoint)

    def new(self) -> AgentAdapter:
        """Return the backend of a new IM session."""
        return AgentAdapter(self)

    async def aclose(self) -> None:
        """Release what the workers' source holds; each adapter closes its own providers."""
        await self.workers.aclose()
