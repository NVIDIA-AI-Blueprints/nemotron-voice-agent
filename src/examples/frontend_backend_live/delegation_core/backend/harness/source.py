# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's source (``backend.adapter: harness``): built once per core, it makes one adapter per IM session."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, ClassVar

from examples.frontend_backend_live.delegation_core.backend.llm.settings import LlmSettings
from examples.frontend_backend_live.delegation_core.backend.llm.source import LlmSource
from examples.frontend_backend_live.delegation_core.backend.port import (
    ControlTools,
    EndpointLookup,
    ToolSource,
    ToolSources,
)
from examples.shared.delegation_contract import ModelEndpoint

from .adapter import HarnessAdapter
from .bridge import McpBridge
from .retirement import Retirements
from .settings import HarnessSettings
from .workdir import WorkFolder

if TYPE_CHECKING:
    from examples.frontend_backend_live.delegation_core.backend.llm.adapter import LlmAdapter

# Optional harness packages, checked before importing their implementations.
GROUP_PACKAGES = ("acp", "mcp")
AUTO_CHOICE = (None, "auto")
PLAIN_TIERS = (None, "auto", "default")


class HarnessSource:
    """An external agent over its protocol: its settings, the model name it serves, and the bridge."""

    name: ClassVar[str] = "harness"

    def __init__(
        self,
        settings: HarnessSettings,
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ):
        """Hold what each IM session's adapter is made of; nothing starts."""
        self.settings, self.endpoints, self.tools, self.vocabulary = settings, endpoints, tools, vocabulary
        self.bridge = McpBridge(settings.bridge, start_timeout_seconds=settings.start_timeout_seconds)
        self.retirements = Retirements()
        self._adapters: set[HarnessAdapter] = set()
        self._closing: asyncio.Task | None = None

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> HarnessSource:
        """Validate ``backend.harness``, load the classes it names and sweep stale work folders; start nothing.

        Raises:
            ValueError: The ``harness`` dependency group is not installed, or the block is invalid.
        """
        if missing := [name for name in GROUP_PACKAGES if find_spec(name) is None]:
            raise ValueError(
                "backend.adapter harness needs the packages of the harness dependency group (missing: "
                f"{', '.join(missing)}): install them with `uv sync --group harness`"
            )
        settings = HarnessSettings.model_validate(block)
        WorkFolder.sweep(settings.work_root)
        return cls(settings, endpoints=endpoints, tools=tools, vocabulary=vocabulary)

    def set_context_budget(self, tokens: int) -> None:
        """Use the Core's configured limit for retained native tool receipts."""
        self.bridge.set_context_budget(tokens)

    def endpoint(self) -> ModelEndpoint:
        """Return the default model endpoint, whose model name the harness serves."""
        return self.endpoints(None)

    def new_restricted(self, tools: Sequence[ToolSource]) -> LlmAdapter:
        """Use the configured backend endpoint through the Core-controlled tool loop."""
        source = LlmSource(LlmSettings(), endpoints=self.endpoints, tools=self.tools)
        return source.new(tools=tools)

    def describe(self) -> dict:
        """Describe what it serves as plain JSON, without credentials or environment values."""
        bridge_only = self.settings.agent.capabilities.tool_mediation == "bridge_only"
        return {
            "adapter": self.name,
            **self.settings.describe(),
            "model": self.endpoint().model,
            "restricted_execution": "native" if bridge_only else "llm",
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` unless ``model`` is the one model name the harness serves."""
        served = self.endpoint().model
        if model != served:
            raise ValueError(f"The harness runs {served!r}; it cannot serve the model {model!r}")

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings an agent with its own model calls cannot honor, as ``(param, message)`` pairs."""
        problems = []
        if settings.get("max_output_tokens") is not None:
            problems.append(("max_output_tokens", "The harness's runtime sets its own output limit."))
        text = settings.get("text") or {}
        if (text.get("format") or {}).get("type", "text") != "text" or text.get("verbosity") is not None:
            problems.append(("text", "The harness's runtime answers in plain text."))
        if settings.get("tool_choice") not in AUTO_CHOICE:
            problems.append(("tool_choice", "The harness's runtime chooses its own tools (tool_choice auto)."))
        hosted = [tool.get("type") for tool in settings.get("tools") or [] if tool.get("type") != "function"]
        problems += [
            ("tools", f"The {kind} tool is not available on the harness: it bridges function tools.")
            for kind in dict.fromkeys(hosted)
        ]
        if (settings.get("reasoning") or {}).get("effort") is not None:
            problems.append(("reasoning", "The harness's runtime sets its own reasoning effort."))
        if settings.get("service_tier") not in PLAIN_TIERS:
            problems.append(("service_tier", f"service_tier {settings['service_tier']!r} is not available here."))
        return problems

    def new(self) -> HarnessAdapter:
        """Return the backend of a new IM session, with a fresh set of its tool sources."""
        if self._closing is not None:
            raise RuntimeError("The harness source is closing")
        adapter = HarnessAdapter(self, tools=self.tools(), on_drained=self._adapters.discard)
        self._adapters.add(adapter)
        return adapter

    async def aclose(self) -> None:
        """Close and join this source's native owners, then its bridge; cancelled callers leave cleanup running."""
        if self._closing is None:
            self._closing = asyncio.create_task(self._close(), name="harness-source-close")
            self._closing.add_done_callback(lambda done: done.cancelled() or done.exception())
        await asyncio.shield(self._closing)

    async def _close(self) -> None:
        adapters = tuple(self._adapters)
        results = []
        try:
            results.extend(await asyncio.gather(*(adapter.close() for adapter in adapters), return_exceptions=True))
            results.extend(
                await asyncio.gather(*(adapter.wait_closed() for adapter in adapters), return_exceptions=True)
            )
        finally:
            try:
                await self.bridge.aclose()
            except BaseException as error:
                results.append(error)
            finally:
                self.retirements.close()
        for result in results:
            if isinstance(result, BaseException):
                raise result
