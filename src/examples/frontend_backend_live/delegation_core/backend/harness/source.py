# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's source (``backend.adapter: harness``): built once per core, it makes one adapter per IM session."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, ClassVar

from examples.frontend_backend_live.delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from examples.shared.delegation_contract import ModelEndpoint

from .adapter import HarnessAdapter
from .bridge import McpBridge
from .settings import HarnessSettings
from .workdir import WorkFolder

# The import names of the packages of the harness dependency group: agent-client-protocol and mcp.
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
                f"backend.adapter harness needs the packages of the harness dependency group (missing: "
                f"{', '.join(missing)}): install them with `uv sync --group harness`"
            )
        settings = HarnessSettings.model_validate(block)
        WorkFolder.sweep(settings.work_root)
        return cls(settings, endpoints=endpoints, tools=tools, vocabulary=vocabulary)

    def endpoint(self) -> ModelEndpoint:
        """Return the default model endpoint, whose model name the harness serves."""
        return self.endpoints(None)

    def describe(self) -> dict:
        """Describe what it serves as plain JSON, without credentials or environment values."""
        return {
            "adapter": self.name,
            **self.settings.describe(),
            "model": self.endpoint().model,
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
        return HarnessAdapter(self, tools=self.tools())

    async def aclose(self) -> None:
        """Stop the bridge (each adapter closes its own runtime processes)."""
        await self.bridge.aclose()
