# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM's source (``backend.adapter: llm``), built once per core; also the default one-shot source."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from examples.frontend_backend_live.delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from examples.frontend_backend_live.delegation_core.backend.providers import Backend, provider_class, provider_for
from examples.shared.delegation_contract import ModelEndpoint

from .adapter import LlmAdapter
from .run import can_clear_up, failed_to_open
from .settings import LlmSettings

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
        """Hold what each IM session's adapter is made of; nothing connects."""
        self.settings, self.endpoints, self.tools = settings, endpoints, tools
        self.make_provider = make_provider or (lambda endpoint: provider_for(endpoint, settings.limits))
        # Checked here, so an unknown API on the default model's entry fails the core's build.
        provider_class(endpoints(None))

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> LlmSource:
        """Validate ``backend.llm`` (``block``) and build the source; a rule conductor reads no ``vocabulary``."""
        return cls(LlmSettings.model_validate(block), endpoints=endpoints, tools=tools)

    def describe(self) -> dict:
        """Return what it serves, as plain JSON with no credentials."""
        default = self.endpoints(None)
        return {
            "adapter": self.name,
            **self.settings.describe(),
            "model": default.model,
            "base_url": default.base_url,
            "api": default.api,
            "max_running": LlmAdapter.max_running,
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if it cannot serve ``model``: the catalog lacks it, or no provider speaks its API."""
        provider_class(self.endpoint(model))

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings the provider of their model cannot honor, as ``(param, message)`` pairs."""
        return provider_class(self.endpoint(settings.get("model"))).unsupported(settings)

    def endpoint(self, model: str | None) -> ModelEndpoint:
        """Return the endpoint ``model`` runs on, by the catalog; ``ValueError`` if the catalog does not list it."""
        try:
            return self.endpoints(model)
        except LookupError:
            raise ValueError(
                f"The backend does not serve the model {model!r}: it is not in the model catalog"
            ) from None

    def new(self) -> LlmAdapter:
        """Return the backend of a new IM session: fresh server tool sources, and a failed round tried again."""
        return LlmAdapter(self, tools=self.tools(), retryable=can_clear_up)

    def new_one_shot(self) -> LlmAdapter:
        """Return the backend of one stateless request: no server tools, and only a round that never opened tried again.

        Its client owns the loop, so a round that opened is never run again for it.
        """
        return LlmAdapter(self, tools=(), retryable=failed_to_open)

    async def aclose(self) -> None:
        """Release nothing: each adapter closes its own providers."""
